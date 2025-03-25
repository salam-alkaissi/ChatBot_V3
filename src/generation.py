# # src/generation.py
# import torch
# # from transformers import T5Tokenizer, T5ForConditionalGeneration
# from src.document_processing import chunk_text
# from transformers import BartTokenizer, BartForConditionalGeneration
# import logging
# import cohere
# import os
# import re
# ##COHERE_API_KEY="ZGzGvuaIHCoMUihq1DoHSpl741wUTwZuXdSkgcKQ"
# ##COHERE_API_KEY="P28PBGM9qfKcd0TJZChtwrhJPSVLK102JcQ2aN0v"

# STRUCTURED_PROMPT_TEMPLATE = """Analyze this document section:
#         {instructions}

#         Section Content:
#         {chunk_text}

#         Required Format:
#         - Bullet points for key concepts
#         - Technical specifications in parentheses
#         - Quantitative metrics emphasized

#         Document section: "{chunk_text}"""

# class SummaryGenerator:
#     def __init__(self):
#         # Set memory management for CUDA
#         # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
#         self.device = torch.device("cpu") #"cuda" if torch.cuda.is_available() else 
#         self.model_name = "facebook/bart-large-cnn" #"google/flan-t5-small"  # Better model for summarization google/flan-t5-base
#         # self.tokenizer = None
#         # self.model = None
#         logging.info(f"Initializing with model: {self.model_name}")
#         self.initialize_model()
#         self.co = cohere.Client(os.getenv("COHERE_API_KEY"))
#         self.chunk_size = 2000
#         # Configure logging
#         # logging.basicConfig(level=logging.INFO)
        
#     def initialize_model(self):
#         """Initialize the BART model and tokenizer."""
#         # try:
#         #     self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=False)
#         #     self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
#         #     self.model = self.model.to(self.device)
#         #     logging.info(f"Loaded model {self.model_name} on {self.device}")
#         # except Exception as e:
#         #     logging.error(f"Model loading failed: {str(e)}")
#         #     raise
#         try:
#             logging.info(f"Attempting to load tokenizer for {self.model_name}")
#             self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
#             logging.info(f"Attempting to load model for {self.model_name}")
#             self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
#             self.model = self.model.to(self.device)
#             logging.info(f"Loaded model {self.model_name} on {self.device}")
#             logging.info(f"Loaded model {self.model_name} on {self.device}")
#         except Exception as e:
#             logging.error(f"Model loading failed: {str(e)}")
#             raise
        
#     def _chunk_text(self, text):
#         """Split text into chunks based on character count."""
#         return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

#     def generate(self, text, max_input_length=1024, max_new_tokens=200):  
#         try:
#             if not text or len(text.strip()) < 50:
#                 return "Insufficient text content for meaningful summary"

#     # Better prompt engineering
#             inputs = self.tokenizer(
#                 f"Generate a comprehensive, detailed summary of the following document: {text}",
#                 max_length=max_input_length,
#                 truncation=True,
#                 padding="max_length",
#                 return_tensors="pt"
#             ).to(self.device)

#             # Enhanced generation parameters
#             outputs = self.model.generate(
#                 inputs.input_ids,
#                 max_new_tokens=max_new_tokens,     # Increased from 150
#                 min_length=50,          # Increased from 50
#                 # length_penalty=2,      # Adjusted for longer summaries
#                 # no_repeat_ngram_size=4,
#                 early_stopping=True,
#                 num_beams=4,
#                 # temperature=0.8,         # Slightly higher for diversity
#                 do_sample=False,         # Better coherence with beam search
#                 repetition_penalty=2.0
#             )

#             return self.postprocess_summary(
#                 self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#             )
    
#         except Exception as e:
#             logging.error(f"Generation failed: {str(e)}")
#             return "Summary generation error"
        

#     def postprocess_summary(self, summary):
#         """Clean up generated summary"""
#         # Remove any bullet points or markdown artifacts
#         summary = summary.replace("•", "").replace("##", "").strip()
        
#         # Capitalize first letter if needed
#         if summary and summary[0].islower():
#             summary = summary[0].upper() + summary[1:]
            
#         # Ensure proper sentence endings
#         if summary and summary[-1] not in {'.', '!', '?'}:
#             summary += '.'
            
#         return summary
    
#     # def generate_structured_summary(self, text):
#         # """Generate structured summary using Cohere"""
#         # try:
#         #     response = self.co.chat(
#         #         message=self._build_cohere_prompt(text),
#         #         model="command-r-plus",
#         #         temperature=0.3,
#         #         preamble="You are a technical documentation analyst",
#         #         connectors=[{"id": "web-search"}]
#         #     )
#         #     return self._format_cohere_response(response.text)
            
#         # except Exception as e:
#         #     return f"Cohere Error: {str(e)}"
#         # """Generate a structured summary using Cohere for the full text."""
#         # try:
#         #     if not self.co:
#         #         raise ValueError("Cohere API key not available")
            
#         #     chunks = self._chunk_text(text)
#         #     all_results = []

#         #     for chunk in chunks:
#         #         prompt = self._build_cohere_prompt(chunk)
#         #         response = self.co.generate(
#         #             model='command-xlarge',
#         #             prompt=prompt,
#         #             max_tokens=500,  # Increased for detailed response
#         #             temperature=0.3,
#         #             stop_sequences=["\n\n"]
#         #         )
#         #         all_results.append(response.generations[0].text)

#         #     return self._synthesize_cohere_results(all_results)
#         # except Exception as e:
#         #     logging.error(f"Cohere structured summary failed: {str(e)}")
#         #     return f"Cohere Error: {str(e)}"
#         # try:
#         #     if not self.co:
#         #         raise ValueError("Cohere API key not available")

#         #     chunks = self._chunk_text(text)
#         #     all_results = []

#         #     for i, chunk in enumerate(chunks):
#         #         prompt = self._build_cohere_prompt(chunk)
#         #         response = self.co.generate(
#         #             model='command-xlarge',
#         #             prompt=prompt,
#         #             max_tokens=500,
#         #             temperature=0.3,
#         #             stop_sequences=["\n\n"]
#         #         )
#         #         all_results.append(response.generations[0].text)

#         #     return self._synthesize_cohere_results(all_results)
#         # except Exception as e:
#         #     logging.error(f"Cohere structured summary failed: {str(e)}")
#         #     return self.generate(text) if text else f"Cohere Error: {str(e)}"
        
#     def _build_cohere_prompt(self, chunk_text):
#         """Build optimized prompt with hardcoded instructions"""
#         if not chunk_text.strip():
#             raise ValueError("Empty chunk text received")

#     # Hardcode instructions to avoid template formatting issues
#     prompt = f"""Analyze this technical document section following these exact requirements:

#     1. Identify 3-5 core innovations with technical specifications
#     2. Extract implementation challenges with quantitative estimates
#     3. Recommend solutions with engineering requirements

#     Format requirements:
#     - Use bullet points with technical details in parentheses
#     - Include numerical values where possible
#     - Mark critical requirements with [CR]

#     Document section:
#     {chunk_text[:4000]}

#     Analysis:"""
    
#     return prompt

#     def generate_structured_summary(self, text):
#         """Optimized Cohere summary generation with error handling"""
#         try:
#             if not self.co:
#                 raise ValueError("Cohere API key not available")

#             chunks = self._chunk_text(text)
#             batch_responses = []
        
#         # Use batch processing if supported by Cohere
#             for chunk in chunks:
#                 prompt = self._build_cohere_prompt(chunk)
#                 response = self.co.generate(
#                     model='command-xlarge',
#                     prompt=prompt,
#                     max_tokens=300,  # Reduced for focused response
#                     temperature=0.2,  # Lower for consistency
#                     frequency_penalty=0.7,
#                     stop_sequences=["\n\nEnd"]
#                 )
#                 batch_responses.append(response.generations[0].text)

#             return self._synthesize_cohere_results(batch_responses)
    
#         except Exception as e:
#             logging.error(f"Structured summary error: {str(e)}")
#             return self.generate(text) if text else "Analysis unavailable"
    
#     def _synthesize_cohere_results(self, results):
#         """Combine chunked Cohere results into a single structured response."""
#         themes = []
#         challenges = []
#         solutions = []
#         impacts = []

#         for result in results:
#             # Use regex to extract sections more robustly
#             theme_match = re.findall(r'1\. Key Themes\s*[\n-]*(.*?)(?=\n2\.|\Z)', result, re.DOTALL)
#             if theme_match:
#                 themes.extend([t.strip() for t in re.findall(r'- (.*)', theme_match[0]) if t.strip()])

#             analysis_match = re.findall(r'2\. Detailed Analysis\s*[\n-]*(.*?)(?=\n3\.|\Z)', result, re.DOTALL)
#             if analysis_match:
#                 challenges_match = re.findall(r'- Technical Challenges\s*[\n-]*(.*?)(?=\n- Solution Approaches|\Z)', analysis_match[0], re.DOTALL)
#                 if challenges_match:
#                     challenges.extend([c.strip() for c in re.findall(r'- (.*)', challenges_match[0]) if c.strip()])
#                 solutions_match = re.findall(r'- Solution Approaches\s*[\n-]*(.*?)(?=\n3\.|\Z)', analysis_match[0], re.DOTALL)
#                 if solutions_match:
#                     solutions.extend([s.strip() for s in re.findall(r'- (.*)', solutions_match[0]) if s.strip()])

#             impact_match = re.findall(r'3\. Impact Assessment\s*[\n-]*(.*?)(?=\n1\.|\Z)', result, re.DOTALL)
#             if impact_match:
#                 impacts.extend([i.strip() for i in re.findall(r'- (.*)', impact_match[0]) if i.strip()])

#         # Limit and deduplicate
#         themes = list(dict.fromkeys(themes))[:5]
#         challenges = list(dict.fromkeys(challenges))[:10]
#         solutions = list(dict.fromkeys(solutions))[:10]
#         impacts = list(dict.fromkeys(impacts))[:10]

#         # Fill with placeholders if sections are empty
#         if not themes:
#             themes = ["- No key themes identified"]
#         if not challenges:
#             challenges = ["- No technical challenges identified"] * 5
#         if not solutions:
#             solutions = ["- No solution approaches identified"] * 5
#         if not impacts:
#             impacts = ["- No impact assessment available."] * 5

#         bullet_points = "\n".join(themes)
#         examples = "\n".join(challenges) + "\n" + "\n".join(solutions)
#         impact_sentences = "\n".join(impacts)

#         return STRUCTURED_PROMPT_TEMPLATE.format(
#             bullet_points=bullet_points,
#             examples=examples,
#             challenges="\n".join(challenges),
#             solutions="\n".join(solutions),
#             impact_sentences=impact_sentences
#         )

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