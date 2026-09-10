import { Component, OnInit } from '@angular/core';
import { ChatService } from '../../services/chat.service';

interface ChatMessage {
  role: 'user' | 'bot';
  content: string;
  mensaje_id?: number;
  calificacion?: number;
}

@Component({
  selector: 'app-coodibot-chat',
  standalone: false,
  templateUrl: './coodibot-chat.component.html',
  styleUrl: './coodibot-chat.component.css'
})
export class CoodibotChatComponent implements OnInit {
  userMessage: string = '';
  isOpen: boolean = false;
  cargando: boolean = false;

  sessionId: string = '';

  chatHistory: ChatMessage[] = [
    { role: 'bot', content: '¡Hola! Soy COODIBOT, ¿cómo te puedo ayudar hoy con el ecosistema de robótica?' }
  ];

  constructor(private chatService: ChatService) { }

  ngOnInit(): void {
    this.sessionId = 'docente_' + Math.random().toString(36).substring(2, 9) + '_' + Date.now();
  }

  toggleChat() {
    this.isOpen = !this.isOpen;
  }

  formatearRespuesta(texto: string): string {
    if (!texto.includes('- Concepto Clave:')) return texto;

    return texto
      .replace(/- Concepto Clave:/g, '<span class="coodi-badge concepto">🧠 Concepto Clave</span>')
      .replace(/- Pasos:/g, '<span class="coodi-badge pasos">⚙️ Pasos</span>')
      .replace(/- OA Vinculado:/g, '<span class="coodi-badge oa">🎯 OA Vinculado</span>')
      .replace(/- Verificación:/g, '<span class="coodi-badge verificacion">✅ Verificación</span>');
  }

  sendMessage() {
    if (!this.userMessage.trim()) return;

    const pregunta = this.userMessage;
    this.chatHistory.push({ role: 'user', content: pregunta });

    this.userMessage = '';
    this.cargando = true;

    // NUEVO: Le pasamos nuestro "sessionId" único al servicio
    this.chatService.enviarMensaje(pregunta, this.sessionId).subscribe({
      next: (res) => {
        const textoConFormato = this.formatearRespuesta(res.respuesta.texto);

        this.chatHistory.push({
          role: 'bot',
          content: textoConFormato,
          mensaje_id: res.respuesta.mensaje_id
        });
        this.cargando = false;
      },
      error: (err) => {
        console.error("Error conectando al backend:", err);
        this.chatHistory.push({
          role: 'bot',
          content: '❌ Error de conexión con el asistente. Verifica que el servidor esté activo.'
        });
        this.cargando = false;
      }
    });
  }

  calificarMensaje(msg: ChatMessage, valor: number) {
    if (!msg.mensaje_id) return;

    msg.calificacion = valor;

    this.chatService.evaluarRespuesta(msg.mensaje_id, valor).subscribe({
      next: (res) => {
        console.log(`Evaluación guardada con éxito: ${res.mensaje}`);
      },
      error: (err) => {
        console.error('Error al enviar la evaluación:', err);
      }
    });
  }
}