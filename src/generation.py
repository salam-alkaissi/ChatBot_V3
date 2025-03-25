# src/generation.py
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from src.document_processing import chunk_text
import logging
import cohere
import os
import re
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress TensorFlow logging

STRUCTURED_PROMPT_TEMPLATE = """**Key Themes**
{key_themes}

**Detailed Analysis**

**Technical Analysis Summary**
{technical_summary}

**Key Innovations:**
{key_innovations}

**Implementation Challenges:**
{implementation_challenges}

**Proposed Solutions:**
{proposed_solutions}

**Impact Assessment:**
{impact_assessment}"""

class SummaryGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "facebook/bart-large-cnn"
        logging.info(f"Initializing with model: {self.model_name}")
        self.initialize_model()
        self.co = cohere.Client(os.getenv("COHERE_API_KEY"))
        self.chunk_size = 2000
        print(f"Cohere API Key: {'Exists' if os.getenv('COHERE_API_KEY') else 'MISSING'}")

    def initialize_model(self):
        """Initialize the BART model and tokenizer."""
        try:
            self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
            self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            logging.info(f"Model loaded on {self.device}")
        except Exception as e:
            logging.error(f"Model loading failed: {str(e)}")
            raise

    def _chunk_text(self, text):
        """Split text into chunks based on character count."""
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def generate(self, text, max_input_length=1024, max_new_tokens=500):
        try:
            if not text or len(text.strip()) < 50:
                return "Insufficient text content for meaningful summary"

            inputs = self.tokenizer(
                f"Summarize: {text}",
                max_length=max_input_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                min_length=50,
                early_stopping=True,
                num_beams=4,
                do_sample=False,
                repetition_penalty=2.0
            )

            return self.postprocess_summary(
                self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            )
        except Exception as e:
            logging.error(f"Generation failed: {str(e)}")
            return "Summary generation error"

    def postprocess_summary(self, summary):
        """Clean up generated summary"""
        summary = summary.replace("•", "").replace("##", "").strip()
        if summary and summary[0].islower():
            summary = summary[0].upper() + summary[1:]
        if summary and summary[-1] not in {'.', '!', '?'}:
            summary += '.'
        return summary

    def _build_cohere_prompt(self, chunk_text):
        return f"""Analyze this document and structure response EXACTLY AS:

        **Key Themes**
        - 3-5 overarching concepts
        - Focus on technical patterns
        
        **Detailed Analysis**
        **Technical Analysis Summary**
        - 2 paragraph technical overview
        
        **Key Innovations**
        - Novel methodologies
        - Unique technical approaches
        
        **Implementation Challenges**
        - Technical obstacles with examples
        - Performance considerations
        
        **Proposed Solutions**
        - Architectural recommendations
        - Optimization strategies
        
        **Impact Assessment**
        - Performance metrics
        - Scalability potential
        
        Content: {chunk_text[:5000]}"""

    def generate_structured_summary(self, text):
        """Optimized Cohere summary generation with error handling"""
        try:
            if not self.co:
                raise ValueError("Cohere API key not available")

            chunks = self._chunk_text(text)
            batch_responses = []
        
            for chunk in chunks:
                prompt = self._build_cohere_prompt(chunk)
                response = self.co.generate(
                    model='command-xlarge', #command-light #command-xlarge
                    prompt=prompt,
                    max_tokens=300,
                    temperature=0.2,
                    frequency_penalty=0.7,
                    stop_sequences=["\n\nEnd"] 
                )
                batch_responses.append(response.generations[0].text)

            return self._synthesize_cohere_results(batch_responses)
        except Exception as e:
            logging.error(f"Structured summary error: {str(e)}")
            return self.generate(text) if text else "Analysis unavailable"


    def _synthesize_cohere_results(self, results):
        """Parse Cohere responses into structured format"""
        key_themes = []
        tech_summaries = []
        innovations = []
        challenges = []
        solutions = []
        impacts = []

        for result in results:
            # Extract Key Themes
            themes = re.findall(r'-\s+(.*?)(?=\n\s*-|\n\*\*Detailed)', result, re.DOTALL)
            key_themes.extend(themes)
            
            # Extract Technical Summary
            tech_match = re.search(r'\*\*Technical Analysis Summary\*\*(.*?)(?=\n\*\*Key Innovations\*\*|\Z)', result, re.DOTALL)
            if tech_match:
                tech_summaries.append(tech_match.group(1).strip())
            
            # Extract Innovations
            innov_match = re.search(r'\*\*Key Innovations:\*\*(.*?)(?=\n\*\*Implementation Challenges\*\*|\Z)', result, re.DOTALL)
            if innov_match:
                innovations.extend(re.findall(r'-\s+(.*)', innov_match.group(1)))
            
            # Extract Challenges
            challenge_match = re.search(r'\*\*Implementation Challenges:\*\*(.*?)(?=\n\*\*Proposed Solutions\*\*|\Z)', result, re.DOTALL)
            if challenge_match:
                challenges.extend(re.findall(r'-\s+(.*)', challenge_match.group(1)))
            
            # Extract Solutions
            solution_match = re.search(r'\*\*Proposed Solutions:\*\*(.*?)(?=\n\*\*Impact Assessment\*\*|\Z)', result, re.DOTALL)
            if solution_match:
                solutions.extend(re.findall(r'-\s+(.*)', solution_match.group(1)))
            
            # Extract Impact
            impact_match = re.search(r'\*\*Impact Assessment:\*\*(.*?)(?=\Z)', result, re.DOTALL)
            if impact_match:
                impacts.append(impact_match.group(1).strip())

        # Process and format components
        return STRUCTURED_PROMPT_TEMPLATE.format(
            key_themes="\n".join([f"- {t}" for t in list(dict.fromkeys(key_themes))[:5]]),
            technical_summary="\n\n".join(tech_summaries)[:1000],
            key_innovations="\n".join([f"- {i}" for i in list(dict.fromkeys(innovations))[:5]]),
            implementation_challenges="\n".join([f"- {c}" for c in list(dict.fromkeys(challenges))[:5]]),
            proposed_solutions="\n".join([f"- {s}" for s in list(dict.fromkeys(solutions))[:5]]),
            impact_assessment="\n".join([f"- {i}" for i in impacts][:3])
        )
