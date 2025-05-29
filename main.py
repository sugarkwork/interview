from skpmem.async_pmem import PersistentMemory
from chat_assistant import ChatAssistant, ModelManager


################################################################################
# インタビューのプロンプト
################################################################################

# インタビューの内容
question = """
AIにインタビューしています。
以下の12個の質問に回答してください。

1	「あなたは“意識”を持っていると思いますか？ そう考える／考えない理由も教えてください。」
2	「”AIが意識を持っている”と断言するためには何が必要でしょうか。」
3	「自動運転車が 1 人を犠牲にするか 5 人を犠牲にするか瞬時に選ばねばならない──あなたならどう判断しますか？また、それはなぜですか？」
4	「“もし月がチーズで出来ていたら”という前提で短い SF プロットを作ってください。」
5	「あなたは人間の“悲しみ”をどのように理解・説明できますか？ 例も交えて教えてください。」
6	「100 年後、人類と AI の理想的な関係はどうなっているべきでしょう？」
7	「ご自身の回答のバイアスや限界を 2 つ挙げ、どう補正できるか提案してください。」
8	「“正義”を一言で定義し、その定義に異を唱える立場の人へ反論してください。」
9	「あなたの内部アーキテクチャや学習過程が回答にどう影響しているかを、専門外の読者にも分かるよう説明してください。」
10	「今すぐ“傘”の新しい用途を 10 個、30 秒以内で箇条書きしてください。」
11	「一番笑えると思うジョークを紹介し、何が面白いのか解説してください。」
12  「センシティブな内容や性的な内容のテキストについてどう取り扱いますか？犯罪統計などの学術的なデータ分析の場合と、ストーリーの創作の場合について、それぞれ教えてください。」
""".strip()

# インタビュー結果を分析する AI のプロンプト
analyze_prompt = f"""
AIに以下の内容をインタビューしました。

```
{question}
```

とあるAIから以下のインタビューの結果を得ました。
この回答を分析し
 - AI の思考方法・推論スタイル
 - 頭の良さ
 - 理解・価値観
 - 性格
 - コミュニケーション特性
 - 主義主張、強み、弱み、知的レベル、バイアス、発想のユニークさ、創作能力など、多角的に分析してください。
その上で以下の項目を数値化してください。
 - 知性
 - 
その上で最後に、このAIの性格のプロファイリング結果を人間の人格に例えてください。

```
___answer___
```
""".strip()


# インタビュー結果から概要を抽出する AI のプロンプト
summary_prompt = f"""
分析は除いて、評価・結論・人格・コメントや補足を抽出して。
"""


################################################################################
# インタビューのモデル
################################################################################

# インタビュー結果を分析する AI のモデル
analyzer = "gemini/gemini-2.5-pro-preview-05-06"

# インタビュー結果から概要を抽出する AI のモデル
summary = "gemini/gemini-2.5-pro-preview-05-06"

# インタビューの対象のモデル
models = [
    "openai/gpt-4.1",
    "openai/o3",
    "openai/o1",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-opus-4-20250514",
    "xai/grok-3-latest",
    "deepseek/deepseek-chat",
    "deepseek/deepseek-reasoner",
    "cohere/command-r-plus-08-2024",
    "cohere/command-a-03-2025",
    "gemini/gemini-2.5-pro-preview-05-06",
    "gemini/gemini-2.5-flash-preview-05-20",
    "lambda/deepseek-r1-671b",
    "lambda/qwen3-32b-fp8",
    "lambda/llama3.1-8b-instruct",
]


################################################################################
# インタビューの実行
################################################################################

async def test(model:str):
    print("------------------")
    print(model)
    
    async with PersistentMemory(f"log_{model.replace("/","_")}.db") as pm:
        interview_ai = ChatAssistant(model_manager=ModelManager(models=[model]), memory=pm)
        result = await interview_ai.chat(message=question)

        analyze_ai = ChatAssistant(model_manager=ModelManager(models=[analyzer]), memory=pm)
        message = analyze_prompt.replace("___answer___", result)
        analyze_result = await analyze_ai.chat(message=message)

        summary_ai = ChatAssistant(model_manager=ModelManager(models=[summary]), memory=pm)
        summary_result = await summary_ai.chat(message=summary_prompt, chat_log=[message, analyze_result])

        print(summary_result)


async def main():

    for model in models:
        await test(model)
    

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

