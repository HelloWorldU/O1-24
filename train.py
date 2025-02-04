from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback, DataCollatorForLanguageModeling
import torch
import numpy as np
import re
import os
import gc
import json
from preprocess import dynamic_padding_collator, load_datasets
from datasets import load_dataset
from functools import partial
from sympy import simplify

os.environ['PYTORCH_CUDA_ALLOC_CONF']="expandable_segments:True,garbage_collection_threshold:0.8"
torch.cuda.empty_cache()
model_path = "/content/drive/MyDrive/O1-24/Qwen2.5-0.5b/cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
tokenizer = AutoTokenizer.from_pretrained(model_path)

def get_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        use_cache=False,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()
    model.config.pad_token_id = tokenizer.pad_token_id
    return model

class MemoryOptimizedTrainer(Trainer):
    def evaluation_loop(self, *args, **kwargs):
        with torch.no_grad():
            torch.cuda.empty_cache()
            output = super().evaluation_loop(*args, **kwargs)
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return output

# 定义评估函数
def compute_metrics(eval_preds):
    predictions = eval_preds.predictions
    labels = eval_preds.label_ids

    correct_count = 0
    total_count = 0

    def _extract_expr(text):
        match = re.search(r"\\boxed{(.*?)=24}", text)
        return match.group(1).replace(" ", "").replace("*", "×").replace("/", "÷") if match else None
    
    for i in range(len(predictions)):
        pred_token_ids = torch.argmax(torch.tensor(predictions[i]), dim=-1)[0].cpu().numpy()
        decoded_pred = tokenizer.decode(pred_token_ids, skip_special_tokens=True)
        label_seq = labels[i][0]
        valid_tokens = [token_id for token_id in label_seq if token_id != -100]
        decoded_label = tokenizer.decode(valid_tokens, skip_special_tokens=True)

        if i < 3:
            print(f"\n样例 {i+1}:")
            print(f"预测输出: {decoded_pred}")
            print(f"正确输出: {decoded_label}")
            print("-" * 50)
            
        pred_expr = _extract_expr(decoded_pred)
        label_expr = _extract_expr(decoded_label)
        
        if pred_expr and label_expr:
            try:
                expr1 = pred_expr.replace("=", "") + "-24"
                expr2 = label_expr.replace("=", "") + "-24"
                is_correct = (simplify(expr1) == 0) and (simplify(expr2) == 0)
            except:
                is_correct = False
            if is_correct:
                correct_count += 1
            elif pred_expr == "No solution found" and label_expr == "No solution found":
                correct_count += 1
        total_count += 1
        
        del pred_token_ids, decoded_pred, decoded_label
        if i % 10 == 0:
            torch.cuda.empty_cache()

    accuracy = correct_count / total_count if total_count > 0 else 0

    print(f"\n=== 评估结果 ===")
    print(f"总样本数: {total_count}")
    print(f"正确样本数: {correct_count}")
    print(f"准确率: {accuracy:.2%}")

    return {
        "accuracy": accuracy,
    }

def _train(processed_train_dataset, processed_validation_dataset):
    model = get_model().to('cuda')
    
    data_collator=partial(dynamic_padding_collator, tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir="/content/drive/MyDrive/O1-24/finetuned_model",
        eval_strategy="steps",
        learning_rate=1e-5,
        eval_accumulation_steps=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        optim="adamw_8bit",
        dataloader_num_workers=0,
        report_to="none",
        include_inputs_for_metrics=False,
        eval_do_concat_batches=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        weight_decay=0.01,
        save_total_limit=2,
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        max_steps=1000,
        warmup_steps=50,
        fp16=False,
        bf16=True,
        tf32=True,
        max_grad_norm=1.0,
        logging_dir="/content/drive/MyDrive/O1-24/finetuned_model/logs",
        gradient_checkpointing=True,
        torch_compile=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train_dataset,
        eval_dataset=processed_validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    class ProgressCallback(TrainerCallback):
      def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:  
          print(f"训练进度: 第 {state.global_step} 步 / 共 {state.max_steps} 步")
          if state.log_history:  
            last_loss = state.log_history[-1].get('loss', None)
            if last_loss:
              print(f"当前损失: {last_loss:.4f}")

    trainer.add_callback(ProgressCallback())
    trainer.train()

    trainer.save_model("/content/drive/MyDrive/O1-24/finetuned_model")
    tokenizer.save_pretrained("/content/drive/MyDrive/O1-24/finetuned_model") 
    

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_dataset = load_dataset('json', data_files='/content/drive/MyDrive/O1-24/train_dataset.json', split='train', streaming=True)
    validation_dataset = load_dataset('json', data_files='/content/drive/MyDrive/O1-24/validation_dataset.json', split='train', streaming=True)

    validation_dataset = validation_dataset.take(100)
    processed_train_dataset, processed_validation_dataset = load_datasets(train_dataset, validation_dataset, tokenizer)

    _train(processed_train_dataset, processed_validation_dataset)