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

  formatearRespuesta(texto: string, oas_vinculados?: any[]): string {
    let textoFormateado = texto;

    if (textoFormateado.includes('- Concepto Clave:')) {
      textoFormateado = textoFormateado
        .replace(/- Concepto Clave:/g, '<span class="coodi-badge concepto">🧠 Concepto Clave</span>')
        .replace(/- Pasos:/g, '<span class="coodi-badge pasos">⚙️ Pasos</span>')
        .replace(/- OA Vinculado:/g, '<span class="coodi-badge oa">🎯 OA Vinculado</span>')
        .replace(/- Verificación:/g, '<span class="coodi-badge verificacion">✅ Verificación</span>');
    }

    if (oas_vinculados && oas_vinculados.length > 0) {
      oas_vinculados.forEach(oa => {
        const html_tooltip = `<span class="coodi-tooltip">${oa.codigo} 👁️<span class="coodi-tooltip-text"><b>${oa.codigo}:</b> ${oa.descripcion}</span></span>`;
        textoFormateado = textoFormateado.replace(oa.codigo, html_tooltip);
      });
    }

    return textoFormateado;
  }

  sendMessage() {
    if (!this.userMessage.trim()) return;

    const pregunta = this.userMessage;
    this.chatHistory.push({ role: 'user', content: pregunta });

    this.userMessage = '';
    this.cargando = true;

    this.chatService.enviarMensaje(pregunta, this.sessionId).subscribe({
      next: (res) => {
        const textoConFormato = this.formatearRespuesta(res.respuesta.texto, res.respuesta.oas_vinculados);

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