import os
import re
import json
import openai
import asyncio
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from azure.identity import CertificateCredential
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Union
import uuid
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("LLM")

os.environ["http_proxy"] = "proxy.jpmchase.net:10443"
os.environ["https_proxy"] = "proxy.jpmchase.net:10443"
if 'no_proxy' in os.environ:
    os.environ["no_proxy"] = os.environ["no_proxy"] + ",jpmchase.net" + ",openai.azure.com"
else:
    os.environ["no_proxy"] = 'localhost,127.0.0.1,jpmchase.net,openai.azure.com'

# compile once only
CLEANR = re.compile('<.*?>')

class BaseLLMCustom:
    def __init__(self, few_shot_df: pd.DataFrame = None, few_shot_count: int = None, model = "gpt-4o-2024-11-20",):
        self.client_id = "0CE1FC06-50EA-4053-ACBD-74577EC25477"
        self.deployment_id = model
        self.tenant_id = "79C738E8-25CD-4C36-ADF6-6EA2ED78F6A4"
        self.base_url = "https://llm-multitenancy-exp.jpmchase.net/ver2/"
        self.scope = "https://cognitiveservices.azure.com/.default"
        self.api_version="2024-10-21"
        self.certificate_path = "/opt/omniai/work/instance1/jupyter/rasa_txsearch/rasa_txsearch/preprocessing/training_data_prep/certs/conversation-analysis-llm-dev.azure.jpmchase.net.PEM"
        self.system_prompt = "You are a helpful banking assistant that always responds in valid JSON format."
        self.few_shot_df = few_shot_df
        self.few_shot_count = few_shot_count
        self.few_shot_examples = None
        self.few_shot_indices = None
        self.access_token = None
        self.client = None
        self.api_key = "34757cc6207e4e999f7a6ae57a502f43"
        
        # Thread pool for parallel execution
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        self.initialize()

    def get_credential(self):
        credential = CertificateCredential(
            client_id=self.client_id,
            certificate_path=self.certificate_path,
            tenant_id=self.tenant_id,
            scope=self.scope,
            logging_enable=False,
        )
        return credential

    def get_access_token(self, verbose=False):
        cred = self.get_credential()
        access_token = cred.get_token(self.scope).token
        if verbose:
            logger.info("===ACCESS_TOKEN:===" + access_token +'\n')
        return access_token

    def create_few_shot_examples(self):
        if self.few_shot_count is None:
            self.few_shot_examples = self.few_shot_df
        else:
            self.few_shot_examples = self.few_shot_df.head(self.few_shot_count) if len(self.few_shot_df) >= self.few_shot_count else self.few_shot_df
        self.few_shot_indices = self.few_shot_examples.index

    def initialize(self, verbose=False):
        """
        Initalize Client

        Args:
            verbose (bool, optional): Prints out Access token. Defaults to True.
        """
        self.get_credential()
        self.access_token = self.get_access_token(verbose=verbose)
        self.client = openai.AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.base_url,
            default_headers={
                "Authorization": f"Bearer {self.access_token}",
                "user_sid": "",
            }
        )
        if self.few_shot_df is not None:
            self.create_few_shot_examples()

    def structure_prompt(self, user_prompt: str, system_prompt: str = None):
        if system_prompt is not None:
            return {"system_message": system_prompt, "user_message": user_prompt}
        else:
            return {"system_message": self.system_prompt, "user_message": user_prompt}

    def create_completion_gpt(self, prompt, max_tokens=1024, temperature=0.1):
        try:
            response = self.client.chat.completions.create(
                model = self.deployment_id,
                messages=[
                    {"role": "system", "content": prompt["system_message"]},
                    {"role": "user", "content": prompt["user_message"]}
                ],
                temperature=temperature,
                max_tokens = max_tokens
            )
            return response
        except Exception as e:
            logger.info(f"Error calling OpenAI API: {e}")
            self.initialize(verbose=False)
            return self.create_completion_gpt(prompt=prompt, max_tokens=max_tokens, temperature=temperature)

    def prompt_completion_gpt(self, user_prompt: str, system_prompt: str = None, temperature: float = 0,
                              max_tokens: int = 1000, verbose: bool = False):
        prompt = self.structure_prompt(user_prompt=user_prompt, system_prompt=system_prompt)
        completion = self.create_completion_gpt(prompt=prompt, temperature=temperature, max_tokens=max_tokens)
        if verbose:
            logger.info(completion.model_dump_json(indent=2))
        if completion.choices != []:
            return completion.choices[0].message.content
        else:
            return None

    def generate_embedding(self, text, model="text-embedding-3-large-1"):
        """Get embedding for a text using OpenAI's embedding API."""
        response = self.client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding

    # Parallel processing methods
    def process_prompts_parallel(self, prompts: List[Dict[str, Any]], temperature: float = 0.1, 
                               max_tokens: int = 1000, max_workers: int = 5) -> List[str]:
        """Process multiple prompts in parallel using threading."""
        if not prompts:
            return []
        
        logger.info(f"Processing {len(prompts)} prompts in parallel with {max_workers} workers...")
        
        def process_single_prompt(prompt_data):
            prompt, index = prompt_data
            try:
                result = self.prompt_completion_gpt(
                    user_prompt=prompt["user_message"],
                    system_prompt=prompt["system_message"],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return index, result
            except Exception as e:
                logger.error(f"Error processing prompt {index}: {e}")
                return index, f"Error: {e}"
        
        # Create indexed prompts
        indexed_prompts = [(prompt, i) for i, prompt in enumerate(prompts)]
        
        # Process in parallel
        results = [None] * len(prompts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(process_single_prompt, prompt_data): i 
                             for i, prompt_data in enumerate(indexed_prompts)}
            
            for future in as_completed(future_to_index):
                try:
                    index, result = future.result()
                    results[index] = result
                except Exception as e:
                    original_index = future_to_index[future]
                    logger.error(f"Error in future for prompt {original_index}: {e}")
                    results[original_index] = f"Error: {e}"
        
        return results

    def choose_processing_method(self, prompts: List[Dict[str, Any]], use_batch_api: bool = False, 
                               batch_threshold: int = 10) -> List[str]:
        """Choose between batch API and parallel processing based on number of prompts."""
        if use_batch_api is None:
            use_batch_api = len(prompts) >= batch_threshold
        
        if use_batch_api and len(prompts) >= 2:
            logger.info(f"Using batch API for {len(prompts)} prompts")
            return self.process_batch_async(prompts)
        else:
            logger.info(f"Using parallel processing for {len(prompts)} prompts")
            return self.process_prompts_parallel(prompts)

    @staticmethod
    def clean_json_response(response_text):
        """
        Cleans and repairs the JSON output from GPT-4 to ensure valid parsing.
        - Removes unwanted triple quotes, extra whitespace, and incorrect characters.
        - Ensures the response strictly follows JSON format.
        """
        if not response_text:
            logger.error("Empty response text provided for JSON cleaning")
            return {"error": "Empty response", "explanation": "No response text provided"}
        
        # Remove markdown artifacts
        cleaned = re.sub(r"```json|```", "", response_text, flags=re.MULTILINE)
        cleaned = cleaned.replace("```json", "").replace("```", "")
        cleaned = cleaned.strip()
        
        # Try parsing as-is first
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Try extracting JSON content between braces
        json_start = cleaned.find('{')
        json_end = cleaned.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            extracted = cleaned[json_start:json_end]
            try:
                return json.loads(extracted)
            except json.JSONDecodeError:
                pass
        
        # Try advanced cleaning
        try:
            # Fix trailing commas
            fixed = re.sub(r',(\s*[}\]])', r'\1', cleaned)
            # Remove problematic characters
            fixed = fixed.replace('\x00', '').replace('\ufeff', '')
            # Try parsing again
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # If all else fails, return the raw response
        logger.warning("⚠️ All JSON parsing strategies failed. Returning raw response.")
        return response_text

    @staticmethod
    def clean_html(raw_html):
        clean_text = re.sub(CLEANR, '', raw_html)
        return clean_text

    def extract_json_from_text(self, text: str) -> str:
        """Extract JSON content from text more reliably."""
        # Try to find JSON content between braces
        matches = JSON_EXTRACT_PATTERN.findall(text)
        if matches:
            # Return the longest match (most likely to be complete)
            return max(matches, key=len)
        
        # Fallback: look for content between first { and last }
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx:end_idx + 1]
        
        return text

    def clean_json_string(self, text: str) -> str:
        """Clean JSON string more aggressively."""
        if not text:
            return text
            
        # Remove markdown code blocks
        text = MARKDOWN_PATTERN.sub('', text)
        
        # Remove HTML tags
        text = CLEANR.sub('', text)
        
        # Remove BOM and other problematic characters
        text = text.replace('\ufeff', '').replace('\x00', '').replace('\r', '')
        
        # Fix common JSON issues
        text = re.sub(r',(\s*[}\]])', r'\1', text)  # Remove trailing commas
        text = re.sub(r'([{,]\s*)(\w+):', r'\1"\2":', text)  # Quote unquoted keys
        
        return text.strip()

    def parse_json_response(self, response_text, clean_response=True):
        if not response_text:
            logger.error("Empty response text provided for parsing")
            return {"error": "Empty response", "explanation": "No response text provided"}
        
        logger.info(f"Parsing JSON response of length {len(response_text)}")
        
        # Strategy 1: Direct parsing
        try:
            response_text = response_text.strip()
            result = json.loads(response_text)
            logger.info("Direct JSON parsing successful")
            return result
        except json.JSONDecodeError as e:
            logger.info(f"Direct parsing failed: {e}")
        
        # Strategy 2: Extract and clean
        try:
            json_content = self.extract_json_from_text(response_text)
            if clean_response:
                json_content = self.clean_json_string(json_content)
            
            result = json.loads(json_content)
            logger.debug("JSON extraction and parsing successful")
            return result
            
        except json.JSONDecodeError as e:
            logger.debug(f"JSON extraction failed: {e}")
        
        # Extract JSON content between first { and last }
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            logger.error("No JSON braces found in response")
            return {"error": "No JSON found", "explanation": "Response does not contain JSON braces"}
        
        json_str = response_text[json_start:json_end]
        logger.info(f"Extracted JSON string of length {len(json_str)}")
        
        if clean_response:
            result = self.clean_json_response(json_str)
            
            # Check if we got a valid dictionary back
            if isinstance(result, dict):
                logger.info("Enhanced JSON parsing successful")
                return result
            elif isinstance(result, str):
                logger.warning("JSON cleaning returned string instead of dict, attempting one more parse")
                try:
                    final_result = json.loads(result)
                    if isinstance(final_result, dict):
                        logger.info("Final parsing attempt successful")
                        return final_result
                except json.JSONDecodeError as e:
                    logger.error(f"Final parsing attempt failed: {e}")
            
            # If we still don't have a dict, return an error
            logger.error("Failed to parse JSON response after all attempts")
            return result
            # return {
            #     "error": "JSON parsing failed", 
            #     "explanation": f"Could not parse response into valid JSON. Length: {len(response_text)}"
            # }
        else:
            return json_str

    def prompt_llm_and_parse_response(self, user_prompt: str, system_prompt: str = None, temperature: float = 0,
                                      max_tokens: int = 1000, response_in_json_expected: bool = True):
        try:
            response_text = self.prompt_completion_gpt(
                user_prompt=user_prompt, 
                system_prompt=system_prompt, 
                temperature=temperature, 
                max_tokens=max_tokens
            )
            
            if not response_text:
                logger.error("No response text received from LLM")
                return {"error": "No response received", "explanation": "Empty response from LLM"}
            
            logger.info(f"Received response from LLM, length: {len(response_text)}")
            
            if response_in_json_expected:
                parsed_response = self.parse_json_response(response_text=response_text)
                
                # Additional validation for expected response structure
                if isinstance(parsed_response, dict):
                    if "error" in parsed_response:
                        logger.error(f"JSON parsing resulted in error: {parsed_response}")
                    else:
                        logger.info("JSON parsing completed successfully")
                    return parsed_response
                else:
                    logger.error(f"JSON parsing returned unexpected type: {type(parsed_response)}")
            return response_text
                
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return {"error": "API Error", "explanation": str(e)}

    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)


if __name__ == "__main__":
    user_prompt = "Tell me about New York City"
    logger.info("\n===USER_PROMPT:===\n", user_prompt, "\n")
    llm = BaseLLMCustom()
    llm.initialize()
    response = llm.prompt_llm_and_parse_response(user_prompt=user_prompt, response_in_json_expected=False)
    logger.info("\n===RESPONSE:===\n", response)
    logger.info("\nDONE!")
