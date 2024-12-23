# QA System Proof of Concept pentru Ol-y
import torch
import torch.nn as nn
import torch.nn.functional as F

class QAProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Procesare context și întrebare
        self.context_encoder = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 2),
            nn.ReLU(),
            nn.Linear(config.n_embd * 2, config.n_embd)
        )
        
        self.question_encoder = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 2),
            nn.ReLU(),
            nn.Linear(config.n_embd * 2, config.n_embd)
        )
        
        # Sistem de atenție pentru găsirea răspunsului
        self.cross_attention = nn.MultiheadAttention(
            config.n_embd, 
            num_heads=8,
            dropout=config.dropout
        )
        
        # Procesare răspuns
        self.answer_processor = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 2),
            nn.ReLU(),
            nn.Linear(config.n_embd * 2, config.n_embd)
        )
        
        # Module de verificare și rafinare
        self.verification = nn.Sequential(
            nn.Linear(config.n_embd * 3, config.n_embd),
            nn.ReLU(),
            nn.Linear(config.n_embd, 1),
            nn.Sigmoid()
        )
        
        self.answer_refiner = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.n_embd,
                nhead=8,
                dim_feedforward=config.n_embd * 4
            ),
            num_layers=2
        )
        
    def process_context(self, context_embeddings):
        """Procesează contextul pentru a extrage informații relevante"""
        return self.context_encoder(context_embeddings)
    
    def process_question(self, question_embeddings):
        """Procesează întrebarea pentru a înțelege intenția"""
        return self.question_encoder(question_embeddings)
    
    def find_answer(self, processed_context, processed_question):
        """Găsește răspunsul folosind atenție între context și întrebare"""
        # Aplicăm cross-attention
        attention_output, attention_weights = self.cross_attention(
            processed_question,
            processed_context,
            processed_context
        )
        
        # Procesăm rezultatul atenției
        answer_candidates = self.answer_processor(attention_output)
        
        return answer_candidates, attention_weights
    
    def verify_answer(self, answer_candidates, processed_context, processed_question):
        """Verifică calitatea și relevanța răspunsului"""
        # Concatenăm toate informațiile relevante
        verification_input = torch.cat([
            answer_candidates,
            processed_context,
            processed_question
        ], dim=-1)
        
        # Calculăm scorul de încredere
        confidence_score = self.verification(verification_input)
        
        return confidence_score
    
    def refine_answer(self, answer_candidates, confidence_score):
        """Rafinează răspunsul bazat pe scorul de încredere"""
        # Aplicăm rafinarea doar pentru răspunsurile cu încredere ridicată
        refined_answer = self.answer_refiner(
            answer_candidates * confidence_score
        )
        
        return refined_answer
    
    def forward(self, context_embeddings, question_embeddings):
        """Forward pass complet pentru sistemul QA"""
        # Procesare inițială
        processed_context = self.process_context(context_embeddings)
        processed_question = self.process_question(question_embeddings)
        
        # Găsire răspuns
        answer_candidates, attention_weights = self.find_answer(
            processed_context, 
            processed_question
        )
        
        # Verificare și rafinare
        confidence_score = self.verify_answer(
            answer_candidates,
            processed_context,
            processed_question
        )
        
        refined_answer = self.refine_answer(answer_candidates, confidence_score)
        
        return {
            'answer': refined_answer,
            'confidence': confidence_score,
            'attention_weights': attention_weights
        }

class AnswerGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Generator de răspunsuri
        self.generator = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.n_embd,
                nhead=8,
                dim_feedforward=config.n_embd * 4
            ),
            num_layers=2
        )
        
        # Proiecție finală pentru vocabular
        self.output_projection = nn.Linear(config.n_embd, config.vocab_size)
        
        # Control pentru diverse aspecte ale generării
        self.temperature_control = nn.Linear(config.n_embd, 1)
        self.length_control = nn.Linear(config.n_embd, 1)
        self.style_control = nn.Linear(config.n_embd, config.n_embd)
        
    def control_generation(self, embedded_answer):
        """Controlează parametrii generării"""
        temperature = torch.sigmoid(self.temperature_control(embedded_answer))
        length_factor = torch.sigmoid(self.length_control(embedded_answer))
        style_vector = self.style_control(embedded_answer)
        
        return temperature, length_factor, style_vector
    
    def forward(self, embedded_answer, max_length=50):
        """Generează răspunsul final"""
        batch_size = embedded_answer.size(0)
        
        # Controlăm parametrii generării
        temperature, length_factor, style_vector = self.control_generation(embedded_answer)
        
        # Ajustăm lungimea maximă bazat pe factorul de lungime
        adjusted_length = int(max_length * length_factor.mean().item())
        
        # Generăm răspunsul token cu token
        generated_sequence = []
        current_input = embedded_answer
        
        for _ in range(adjusted_length):
            # Aplicăm decoder-ul
            decoder_output = self.generator(
                current_input,
                embedded_answer
            )
            
            # Aplicăm stilul
            styled_output = decoder_output + style_vector
            
            # Proiectăm pe vocabular
            logits = self.output_projection(styled_output)
            
            # Aplicăm temperatura
            scaled_logits = logits / temperature
            
            # Sampling
            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs[:, -1, :], 1)
            
            generated_sequence.append(next_token)
            
            # Pregătim input-ul pentru următorul pas
            next_token_embedded = F.embedding(
                next_token,
                self.output_projection.weight.transpose(0, 1)
            )
            current_input = torch.cat([current_input, next_token_embedded], dim=1)
        
        return torch.cat(generated_sequence, dim=1)

class OLyQASystem(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Componentele principale
        self.qa_processor = QAProcessor(config)
        self.answer_generator = AnswerGenerator(config)
        
        # Integrare cu sistemul emoțional
        self.emotional_modulation = nn.Linear(config.n_embd, 64)  # Pentru 64 emoții
        
    def modulate_with_emotions(self, answer_embedding, emotion_state):
        """Modulează răspunsul cu starea emoțională"""
        emotion_factors = torch.sigmoid(self.emotional_modulation(answer_embedding))
        modulated_answer = answer_embedding * emotion_factors.unsqueeze(-1)
        return modulated_answer
    
    def forward(self, context, question, emotion_state=None):
        """Forward pass complet pentru sistemul QA"""
        # Procesăm întrebarea și găsim răspunsul
        qa_output = self.qa_processor(context, question)
        
        # Modulăm cu emoții dacă sunt disponibile
        if emotion_state is not None:
            qa_output['answer'] = self.modulate_with_emotions(
                qa_output['answer'],
                emotion_state
            )
        
        # Generăm răspunsul final
        generated_answer = self.answer_generator(qa_output['answer'])
        
        return {
            'answer_tokens': generated_answer,
            'confidence': qa_output['confidence'],
            'attention_pattern': qa_output['attention_weights']
        }

# Funcții de utilitate pentru testare

def test_qa_system():
    """Funcție pentru testarea sistemului QA"""
    # Configurație de test
    class TestConfig:
        n_embd = 768
        vocab_size = 50257
        dropout = 0.1
    
    config = TestConfig()
    
    # Instanțiem sistemul
    qa_system = OLyQASystem(config)
    
    # Date de test
    batch_size = 2
    seq_length = 10
    
    context = torch.randn(batch_size, seq_length, config.n_embd)
    question = torch.randn(batch_size, seq_length, config.n_embd)
    emotion_state = torch.randn(batch_size, 64)
    
    # Rulăm testul
    with torch.no_grad():
        output = qa_system(context, question, emotion_state)
    
    print("Test completat cu succes!")
    print(f"Forma răspuns: {output['answer_tokens'].shape}")
    print(f"Scor încredere mediu: {output['confidence'].mean().item():.4f}")

if __name__ == "__main__":
    test_qa_system()


################################################### v2 ###############################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from IPython.display import clear_output
import pandas as pd

class SimpleQASystem(nn.Module):
    def __init__(self, vocab_size=1000, embedding_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        # Embeddings pentru tokens
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Encoder pentru întrebare și context
        self.encoder = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1
        )
        
        # Atenție pentru găsirea răspunsului
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # Proiecție finală
        self.output = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, question, context):
        # Embeddings
        q_emb = self.embeddings(question)
        c_emb = self.embeddings(context)
        
        # Encoding
        q_enc = self.encoder(q_emb)
        c_enc = self.encoder(c_emb)
        
        # Attention între întrebare și context
        attn_output, attn_weights = self.attention(
            query=q_enc.transpose(0, 1),
            key=c_enc.transpose(0, 1),
            value=c_enc.transpose(0, 1)
        )
        
        # Generare răspuns
        logits = self.output(attn_output.transpose(0, 1))
        
        return logits, attn_weights

def generate_test_data(batch_size=4, seq_len=10, vocab_size=1000):
    question = torch.randint(0, vocab_size, (batch_size, seq_len))
    context = torch.randint(0, vocab_size, (batch_size, seq_len))
    return question, context

class VisualizationTracker:
    def __init__(self):
        self.losses = []
        self.attention_patterns = []
        self.token_distributions = []
        self.confidence_scores = []
        
    def update(self, loss, attention, logits):
        self.losses.append(loss)
        self.attention_patterns.append(attention.detach().cpu().numpy())
        
        # Calculăm distribuția token-urilor
        token_probs = F.softmax(logits, dim=-1)
        self.token_distributions.append(token_probs.mean(dim=1).detach().cpu().numpy())
        
        # Calculăm scoruri de încredere
        confidence = token_probs.max(dim=-1)[0].mean().item()
        self.confidence_scores.append(confidence)
    
    def plot_training_progress(self, clear=True):
        if clear:
            clear_output(wait=True)
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Plot pentru loss
        ax1 = plt.subplot(2, 2, 1)
        plt.plot(self.losses, 'b-', label='Training Loss')
        plt.title('Evoluție Loss în Timp')
        plt.xlabel('Pas de Training')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # 2. Heatmap pentru ultimul pattern de atenție
        ax2 = plt.subplot(2, 2, 2)
        last_attention = self.attention_patterns[-1][0]  # Primul head
        sns.heatmap(last_attention[:5, :5], 
                   annot=True, 
                   fmt='.3f', 
                   cmap='viridis',
                   ax=ax2)
        plt.title('Pattern Atenție Recent')
        plt.xlabel('Poziție Context')
        plt.ylabel('Poziție Query')
        
        # 3. Distribuția confidenței modelului
        ax3 = plt.subplot(2, 2, 3)
        plt.plot(self.confidence_scores, 'g-', label='Confidence Score')
        plt.title('Evoluție Confidence Score')
        plt.xlabel('Pas de Training')
        plt.ylabel('Confidence')
        plt.grid(True)
        plt.legend()
        
        # 4. Distribuția token-urilor
        ax4 = plt.subplot(2, 2, 4)
        recent_dist = self.token_distributions[-1][0]
        plt.hist(recent_dist, bins=50, alpha=0.75)
        plt.title('Distribuția Probabilităților Token-urilor')
        plt.xlabel('Probabilitate')
        plt.ylabel('Frecvență')
        
        plt.tight_layout()
        plt.show()
    
    def plot_attention_evolution(self):
        """Vizualizează evoluția pattern-urilor de atenție"""
        n_steps = min(5, len(self.attention_patterns))
        fig, axes = plt.subplots(1, n_steps, figsize=(20, 4))
        
        for i, ax in enumerate(axes):
            idx = i * (len(self.attention_patterns) // n_steps)
            attention = self.attention_patterns[idx][0]
            sns.heatmap(attention[:5, :5], 
                       annot=True, 
                       fmt='.3f', 
                       cmap='viridis',
                       ax=ax)
            ax.set_title(f'Pas {idx}')
            
        plt.suptitle('Evoluția Pattern-urilor de Atenție', y=1.05)
        plt.tight_layout()
        plt.show()
    
    def plot_detailed_metrics(self):
        """Vizualizează metrici detaliate despre performanța modelului"""
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Distribuția valorilor de atenție
        ax1 = plt.subplot(2, 2, 1)
        all_attention_values = np.concatenate([a.flatten() for a in self.attention_patterns])
        sns.histplot(all_attention_values, bins=50, ax=ax1)
        plt.title('Distribuția Valorilor de Atenție')
        plt.xlabel('Valoare Atenție')
        plt.ylabel('Frecvență')
        
        # 2. Analiza stabilității loss-ului
        ax2 = plt.subplot(2, 2, 2)
        losses_diff = np.diff(self.losses)
        plt.plot(losses_diff, 'r-', label='Diferență Loss')
        plt.title('Stabilitatea Loss-ului (Diferențe între Pași)')
        plt.xlabel('Pas')
        plt.ylabel('Diferență Loss')
        plt.grid(True)
        plt.legend()
        
        # 3. Heatmap corelație între metrici
        ax3 = plt.subplot(2, 2, 3)
        metrics_df = pd.DataFrame({
            'Loss': self.losses,
            'Confidence': self.confidence_scores
        })
        sns.heatmap(metrics_df.corr(), annot=True, cmap='coolwarm', ax=ax3)
        plt.title('Corelație între Metrici')
        
        # 4. Box plot pentru distribuția attention weights
        ax4 = plt.subplot(2, 2, 4)
        attention_data = [p[0].flatten() for p in self.attention_patterns[-5:]]
        plt.boxplot(attention_data)
        plt.title('Distribuția Attention Weights (Ultimii 5 Pași)')
        plt.xlabel('Pas')
        plt.ylabel('Valoare')
        
        plt.tight_layout()
        plt.show()

def run_qa_test_with_viz():
    print("Inițializare model QA simplu și tracker vizualizări...")
    
    VOCAB_SIZE = 1000
    EMBEDDING_DIM = 256
    BATCH_SIZE = 4
    SEQ_LEN = 10
    NUM_EPOCHS = 5
    
    model = SimpleQASystem(VOCAB_SIZE, EMBEDDING_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    viz_tracker = VisualizationTracker()
    
    print("\nÎncepere training cu vizualizări...")
    for epoch in range(NUM_EPOCHS):
        question, context = generate_test_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        
        # Training step
        optimizer.zero_grad()
        logits, attention = model(question, context)
        target = torch.randint(0, VOCAB_SIZE, (logits.size(0), logits.size(1)))
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), target.view(-1))
        loss.backward()
        optimizer.step()
        
        # Update vizualizări
        viz_tracker.update(loss.item(), attention, logits)
        viz_tracker.plot_training_progress()
        
        print(f"Epoca {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}")
    
    print("\nGenerare vizualizări finale...")
    viz_tracker.plot_training_progress()
    viz_tracker.plot_attention_evolution()
    viz_tracker.plot_detailed_metrics()
    
    return model, viz_tracker

if __name__ == "__main__":
    model, viz_tracker = run_qa_test_with_viz()