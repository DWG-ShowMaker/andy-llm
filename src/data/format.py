def format_dialogue(example):
    """将数据格式化为统一的对话格式"""
    if isinstance(example, str):
        return example
        
    # Belle 格式
    if "instruction" in example and "response" in example:
        return f"Human: {example['instruction']}\nAssistant: {example['response']}"
        
    # Alpaca 格式
    if "input" in example and "output" in example:
        return f"Human: {example['input']}\nAssistant: {example['output']}"
        
    # ChatGPT 格式
    if "conversation" in example:
        dialogue = []
        for turn in example["conversation"]:
            if turn["role"] == "user":
                dialogue.append(f"Human: {turn['content']}")
            elif turn["role"] == "assistant":
                dialogue.append(f"Assistant: {turn['content']}")
        return "\n".join(dialogue)
        
    # QA 格式
    if "question" in example and "answer" in example:
        return f"Human: {example['question']}\nAssistant: {example['answer']}"
        
    return example[config.text_column] 