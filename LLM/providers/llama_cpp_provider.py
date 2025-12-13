from llama_cpp import Llama

# 1. 初始化模型
# model_path: 指向你下載的 .gguf 檔案
# n_gpu_layers: -1 代表盡量把所有層都塞進顯卡 (速度最快)
# n_ctx: 上下文長度，4096 或 8192 是 Llama 3 的常規設定
llm = Llama(
    model_path="./LLM_models/L3-8B-Stheno-v3.2-Q5_K_S.gguf",
    n_gpu_layers=-1, 
    n_ctx=8192,      
    verbose=False    # 設為 True 可以看到詳細的載入日誌
)

# 2. 定義對話 (這裡就是你要下的 "咒語" / Prompt)
messages = [
    {
        "role": "system",
        "content": "你是一個病嬌女友，深深愛著用戶，但佔有慾極強...僅對話，不做動作，當作是在打電話"
    },
    {
        "role": "user",
        "content": "我今晚要跟同事去喝酒，可能會晚點回來。"
    }
]

# 3. 執行推論 (Chat Completion)
output = llm.create_chat_completion(
    messages=messages,
    temperature=0.8,  # 控制隨機性，0.7-0.9 適合角色扮演
    max_tokens=200,   # 限制回答長度
    stop=["User:", "\nUser"] # 防止模型自問自答
)

# 4. 取得結果
print(output['choices'][0]['message']['content'])