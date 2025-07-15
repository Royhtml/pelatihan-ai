import torch
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    BertForQuestionAnswering,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    pipeline,
    AutoModelForCausalLM
)
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Dict
import random

class EnhancedChatBot:
    def __init__(self):
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load classification model for intent recognition
        self.intent_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.intent_model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', 
            num_labels=5  # More intent categories
        )
        self.intent_model.to(self.device)
        
        # Load QA model for factual questions
        self.qa_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.qa_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.qa_model.to(self.device)
        
        # Load generative model (using GPT-2 for better generation)
        self.generative_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.generative_tokenizer.pad_token = self.generative_tokenizer.eos_token
        self.generative_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.generative_model.to(self.device)
        
        # Context for QA
        self.context = """
        Our company was founded in 2010. We specialize in AI technology. 
        Our headquarters is in Jakarta. We have 50 employees.
        The weather today is sunny. Our CEO is John Doe.
        We offer products in natural language processing, computer vision, and data analytics.
        Our main competitors are TechAI and NeuralSolutions.
        """
        
        # Conversation history for context
        self.conversation_history = []
        self.max_history = 5  # Keep last 5 exchanges
        
        # Intent labels and responses
        self.intent_responses = {
            0: ["Hello! How can I assist you today?", "Hi there! What can I do for you?", "Greetings! How may I help?"],
            1: ["Goodbye! Have a great day!", "See you later!", "Bye! Come back if you have more questions."],
            2: ["I'd be happy to help with that.", "Let me provide some information about that.", "Here's what I know about that topic..."],
            3: ["I'm not sure I understand. Could you rephrase that?", "Could you clarify what you mean?", "I didn't quite get that."],
            4: ["That's an interesting point.", "Thanks for sharing that.", "I appreciate your input."]
        }
        
        # Train the intent classifier
        self.train_intent_classifier()
    
    def train_intent_classifier(self):
        # Expanded training data for better intent recognition
        data = {
            "text": [
                # Greetings
                "Hello", "Hi there", "Good morning", "Hey", "What's up", 
                # Goodbyes
                "Goodbye", "See you later", "Bye", "Farewell", "Take care",
                # Questions/Info requests
                "What can you tell me about", "I need information about", "Explain", "Tell me about",
                # Confusion
                "I don't understand", "What does this mean", "That doesn't make sense",
                # Statements
                "That's interesting", "I think that", "In my opinion"
            ],
            "label": [
                0, 0, 0, 0, 0,  # greetings
                1, 1, 1, 1, 1,  # goodbyes
                2, 2, 2, 2,      # info requests
                3, 3, 3,         # confusion
                4, 4, 4          # statements
            ]
        }
        
        df = pd.DataFrame(data)
        
        # Create dataloader
        dataset = ChatDataset(
            texts=df.text.to_numpy(),
            labels=df.label.to_numpy(),
            tokenizer=self.intent_tokenizer,
            max_len=64
        )
        data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Train with learning rate scheduling
        optimizer = AdamW(self.intent_model.parameters(), lr=2e-5, correct_bias=False)
        total_steps = len(data_loader) * 3
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        for epoch in range(3):
            self.train_epoch(self.intent_model, data_loader, optimizer, scheduler, self.device)
    
    def train_epoch(self, model, data_loader, optimizer, scheduler, device):
        model.train()
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    
    def classify_intent(self, text: str) -> int:
        """Classify user input into intent categories"""
        encoding = self.intent_tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=64,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.intent_model(input_ids=input_ids, attention_mask=attention_mask)
        
        _, prediction = torch.max(outputs.logits, dim=1)
        return prediction.item()
    
    def answer_question(self, question: str) -> str:
        """Answer factual questions using the QA model"""
        try:
            inputs = self.qa_tokenizer(
                question,
                self.context,
                add_special_tokens=True,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            with torch.no_grad():
                outputs = self.qa_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            answer = self.qa_tokenizer.convert_tokens_to_string(
                self.qa_tokenizer.convert_ids_to_tokens(
                    input_ids[0][answer_start:answer_end]
                )
            )
            
            return answer if answer else "I don't have information about that."
        except:
            return "I encountered an error processing your question."
    
    def generate_response(self, prompt: str, max_length: int = 100) -> str:
        """Generate a conversational response using GPT-2"""
        # Combine conversation history with new prompt
        full_prompt = "\n".join(self.conversation_history[-self.max_history:] + f"\nUser: {prompt}\nBot:")
        
        inputs = self.generative_tokenizer.encode(
            full_prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        # Generate more natural-sounding responses
        outputs = self.generative_model.generate(
            inputs,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=self.generative_tokenizer.eos_token_id
        )
        
        response = self.generative_tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Extract only the new response part
        response = response[len(full_prompt):].strip()
        return response.split("\n")[0]  # Return only the first line
    
    def get_response(self, user_input: str) -> str:
        """Determine the best response strategy based on user input"""
        # Update conversation history
        self.conversation_history.append(f"User: {user_input}")
        
        # Clean input
        clean_input = user_input.strip().lower()
        
        # Check for exit conditions
        if any(word in clean_input for word in ["bye", "goodbye", "exit", "quit"]):
            response = random.choice(self.intent_responses[1])
            self.conversation_history.append(f"Bot: {response}")
            return response
        
        # Check if it's a clear question
        if "?" in user_input or any(word in clean_input for word in ["what", "how", "when", "where", "why", "who"]):
                answer = self.answer_question(user_input)
                if answer.lower() not in ["", "i don't know", "i don't have information about that"]:
                    self.conversation_history.append(f"Bot: {answer}")
                    return answer
        
        # Classify intent
        intent = self.classify_intent(user_input)
        
        # For greetings, use predefined responses
        if intent == 0:  # greeting
            response = random.choice(self.intent_responses[0])
        elif intent == 1:  # goodbye
            response = random.choice(self.intent_responses[1])
        elif intent == 2:  # info request
            # Try to answer with QA first, then fall back to generation
            answer = self.answer_question(user_input)
            if answer.lower() not in ["", "i don't know", "i don't have information about that"]:
                response = answer
            else:
                response = self.generate_response(user_input)
        else:
            # For other cases, generate a response
            response = self.generate_response(user_input)
        
        # Update conversation history
        self.conversation_history.append(f"Bot: {response}")
        return response
    
    def chat(self):
        """Start interactive chat session"""
        print("\nEnhanced BERT Chatbot (ChatGPT-like)")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.\n")
        print(random.choice(self.intent_responses[0]))
        
        while True:
            try:
                user_input = input("You: ")
                if not user_input.strip():
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print(f"Bot: {random.choice(self.intent_responses[1])}")
                    break
                
                response = self.get_response(user_input)
                print(f"Bot: {response}")
                
            except KeyboardInterrupt:
                print("\nBot: Goodbye!")
                break
            except Exception as e:
                print("Bot: Sorry, I encountered an error. Could you rephrase that?")
                print(f"[Debug: {str(e)}]")

class ChatDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

if __name__ == "__main__":
    chatbot = EnhancedChatBot()
    chatbot.chat()