import os
import json
import requests
from dotenv import load_dotenv
from groq import Groq
import re
import time
import argparse

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")

HUGGINGFACE_KEY = os.getenv("HUGGINGFACE_API_KEY")
HEADERS = {"X-HuggingFace-Api-Key": HUGGINGFACE_KEY}

COMPLETENESS_PROMPT = os.getenv("COMPLETENESS_PROMPT")


def try_parse_json(text):
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, str(e)


class ResponseEvaluator:

    def __init__(self):
        self.completeness_prompt = os.getenv("COMPLETENESS_PROMPT")
        self.eval_model = "qwen/qwen3-32b"


    def evaluate(self, query, response, essential_info, feature):
        try:
            client = Groq(api_key=GROQ_API_KEY)
            prompt = self.completeness_prompt

            bullets = "\n- ".join(essential_info)                
            message = f"""---
                Alzheimer's Disease-related query: {query}

                Essential elements expected in a relevant response:
                - {bullets}

                Evaluate the following response: 
                {response}
            """

            chat_completion = client.chat.completions.create(
                model=self.eval_model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )

            json_obj, error = try_parse_json(chat_completion.choices[0].message.content)
            while type(error) == str:
                prompt = "Your job is to extract the relevant information from the previous response and output it as a JSON object **only**, in exactly the following format: {\"score\": 1|2|3|4|5,\"reasoning\": \"Maximum of three sentences giving reasoning to the score.\"}. Do not include any other text, explanations, or markdown. The JSON must be valid and parsable."
                chat_completion = client.chat.completions.create(
                    model=self.eval_model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": json_obj}
                    ],
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                json_obj, error = try_parse_json(chat_completion.choices[0].message.content)

            
            client.close()

            return {
                "response": json.dumps(response),
                "evaluation": {feature: json_obj}
            }
        
        except Exception as e:
            print("Error during evaluation:", e)
            return "{}"


def general_llm_generation(model, question):
    """Query a general-purpose LLM without retrieval context."""
    query = question["question"]
    client = Groq(api_key=GROQ_API_KEY)
    general_system_msg = """
    You are an information system that answers questions about Alzheimer's disease using your knowledge.
    Your task is to generate a complete, detailed, and informative response by synthesizing relevant information from your training. 
    You may combine facts from different aspects of the topic to provide a comprehensive answer.
    Always answer in your own words using clear and precise language. Avoid speculation beyond established medical knowledge.
    If you are uncertain about specific details or if the question is beyond your knowledge, clearly state your limitations.
    """
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": general_system_msg},
            {"role": "user", "content": query}
        ]
    )
    client.close()
    return completion.choices[0].message.content 


def perfect_rag_generation(model, question):
    """Generation of a RAG application assuming perfect retrieval of relevant context."""
    
    system_msg = """
    You are an information system that answers questions about Alzheimer's disease using only the information provided in the context.
    Your task is to generate a complete, detailed, and informative response by synthesizing as much relevant information as possible from the context. 
    You may combine facts from different parts of the context, but do not add anything that is not explicitly stated.
    Always answer in your own words using clear and precise language. Do not copy text verbatim unless necessary for accuracy. Avoid speculation.
    If the answer cannot be derived from the context, respond with: "The answer is not available in the provided context."
    Do not mention the context or refer to it in your answer.
    """

    context = "\n\n".join(question["essential_chunks"])
    user_prompt = f"""
    Context:
    {context}

    Question:
    {question["question"]}
    """

    client = Groq(api_key=GROQ_API_KEY)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ]
    )

    client.close()
    return completion.choices[0].message.content

    
def rag_pipeline_generation(model, query, collections):
    """
    Evaluation of RAG Application on single query. Pass query to existing FastAPI endpoing to generate a chat.
    """
    chat = {"role": "user", "content": query}

    url = f"{API_BASE_URL}/generate-chat?"
    response = requests.post(url, json={"model": model, "collections": collections, "chats": [chat], "limit": 2})

    if response.status_code == 200:
        result = response.json()
        return result["response"]
    

def main():

    parser = argparse.ArgumentParser(description="Evaluating LLMs in RAG Applications")
    parser.add_argument("--generate", action="store_true", help="Generate model responses.")
    parser.add_argument("--evaluate", type=str, help="Evaluate saved responses.")
    args = parser.parse_args()

    try:
        if args.generate:
            models = [
                "gemma2-9b-it", 
                "llama-3.1-8b-instant", 
                "llama-3.3-70b-versatile",
                "meta-llama/llama-4-maverick-17b-128e-instruct",
                "qwen/qwen3-32b"
            ]

            with open("questions/questions.json") as file:
                question_set = json.load(file)

            for question in question_set:
                raw_question = question["question"]

                safe_filename = re.sub(r'\W+', '_', raw_question)  
                safe_filename = safe_filename.strip('_')  
                filename = safe_filename.lower()

                responses = []
                for model in models:
                    response = {
                        "model": model,
                        "general_llm": general_llm_generation(model, question),
                        "perfect_rag": perfect_rag_generation(model, question),
                        "rag_pipeline": rag_pipeline_generation(model, question["question"], "Articles")
                    }
                    responses.append(response)
                    time.sleep(10)
                
                os.makedirs(name="responses", exist_ok=True)

                with open(f"responses/{filename}.json", "w") as f:
                    json.dump({
                        "question": question["question"],
                        "category": question["category"],
                        "essential_info": question["essential_info"],
                        "responses": responses
                    }, f, indent=4)
        

        if args.evaluate:
            evaluator = ResponseEvaluator()
            for file in os.listdir("responses"):
                with open(os.path.join("responses", file)) as f:
                    data = json.load(f)
                
                evaluations = []
                for response in data["responses"]:
                    general_llm = response["general_llm"]
                    perfect_rag = response["perfect_rag"]
                    rag_pipeline = response["rag_pipeline"]

                    general_llm_eval = evaluator.evaluate(data["question"], general_llm, data["essential_info"], args.evaluate)
                    perfect_rag_eval = evaluator.evaluate(data["question"], perfect_rag, data["essential_info"], args.evaluate)
                    rag_pipeline_eval = evaluator.evaluate(data["question"], rag_pipeline, data["essential_info"], args.evaluate)
                    
                    evaluations.append({
                        "model": response["model"],
                        "general_llm_eval": general_llm_eval,
                        "perfect_rag_eval": perfect_rag_eval,
                        "rag_pipeline_eval": rag_pipeline_eval
                    })

                    time.sleep(5)

                os.makedirs(name="evaluations2", exist_ok=True)
                with open(f"evaluations2/{file.replace('.json', '_eval.json')}", "w") as f:
                    json.dump({
                        "question": data["question"],
                        "category": data["category"],
                        "evaluations": evaluations
                    }, f, indent=4)


    except Exception as e:
        print(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()