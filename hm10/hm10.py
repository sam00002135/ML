import pandas as pd
import ollama

df = pd.read_csv("data.csv")

sample = df.head(10).to_string()

prompt = f"""Here is a sample of a dataset:\n{sample}\n
Please help analyze the data. 
- What patterns do you see?
- What questions can we ask about it?
- Any obvious insights or issues?
"""

response = ollama.chat(
    model="llama3:8b",
    messages=[{"role": "user", "content": prompt}]
)

print("\n📊 AI Analysis:\n")
print(response['message']['content'])
