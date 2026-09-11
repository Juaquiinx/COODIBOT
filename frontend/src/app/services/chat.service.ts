import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

// Conecta el chat y el panel admin con los endpoints del backend (main.py)
@Injectable({
  providedIn: 'root'
})
export class ChatService {
  private apiUrl = 'http://localhost:8000/api/chat';

  constructor(private http: HttpClient) { }

  enviarMensaje(pregunta: string, sessionId: string = 'sesion_docente_default'): Observable<any> {
    const body = {
      pregunta: pregunta,
      session_id: sessionId
    };
    return this.http.post<any>(this.apiUrl, body);
  }

  evaluarRespuesta(mensajeId: number, calificacion: number): Observable<any> {
    const body = {
      mensaje_id: mensajeId,
      calificacion: calificacion
    };
    return this.http.put<any>(`${this.apiUrl}/evaluar`, body);
  }



  obtenerRespuestasMalas(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl.replace('/chat', '/admin/malas')}`);
  }

  corregirOA(mensajeId: number, pineconeIds: string, nuevoOA: string): Observable<any> {
    const body = {
      mensaje_id: mensajeId,
      pinecone_ids: pineconeIds,
      nuevo_oa: nuevoOA
    };
    return this.http.put<any>(`${this.apiUrl.replace('/chat', '/admin/corregir-oa')}`, body);
  }


}

