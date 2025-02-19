import json

def format_dialogue(example):
    """格式化对话数据
    
    Args:
        example: 包含 system 和 conversation 字段的数据样本
        
    Returns:
        格式化后的样本
    """
    try:
        # 如果 conversation 是字符串，需要解析成 JSON
        if isinstance(example['conversation'], str):
            conversation = json.loads(example['conversation'])
        else:
            conversation = example['conversation']
            
        # 构建格式化的对话文本
        formatted_text = f"<system>{example['system']}</system>\n"
        
        for turn in conversation:
            if 'human' in turn:
                formatted_text += f"<human>{turn['human']}</human>\n"
            if 'assistant' in turn:
                formatted_text += f"<assistant>{turn['assistant']}</assistant>\n"
                
        return {
            'text': formatted_text,
            'system': example['system'],
            'conversation': example['conversation']
        }
    except Exception as e:
        print(f"格式化对话时出错: {str(e)}")
        return example

def format_dialogue_old(example):
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