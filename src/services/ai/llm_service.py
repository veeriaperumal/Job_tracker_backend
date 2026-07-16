from litellm import acompletion 
from src.config.settings import settings
from src.prompt.system_prompt import SYSTEM_PROMPT


async def chat(model:str,user:str):
    response = await acompletion(
        model=model,
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":user},
        ],
        stream=True,
        temperature=0.7,
        max_tokens=100
    )
    async for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta