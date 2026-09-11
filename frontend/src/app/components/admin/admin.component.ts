import { Component, OnInit } from '@angular/core';
import { ChatService } from '../../services/chat.service';

@Component({
  selector: 'app-admin',
  standalone: false,
  templateUrl: './admin.component.html',
  styleUrl: './admin.component.css'
})
export class AdminComponent implements OnInit {
  respuestasMalas: any[] = [];
  cargando: boolean = false;

  constructor(private chatService: ChatService) { }

  ngOnInit(): void {
    this.cargarMalas();
  }

  // Trae del backend las respuestas que los docentes calificaron negativo
  cargarMalas() {
    this.chatService.obtenerRespuestasMalas().subscribe({
      next: (res) => {
        if (res.estado === 'éxito') {

          this.respuestasMalas = res.data.map((item: any) => ({ ...item, nuevo_oa: '' }));
        }
      },
      error: (err) => console.error('Error al cargar respuestas malas:', err)
    });
  }

  // Envia el OA corregido para actualizar el metadato del vector en Pinecone
  guardarCorreccion(item: any) {
    if (!item.nuevo_oa.trim()) {
      alert("Debes ingresar un código OA válido.");
      return;
    }

    this.cargando = true;
    this.chatService.corregirOA(item.mensaje_id, item.pinecone_ids, item.nuevo_oa).subscribe({
      next: (res) => {
        if (res.estado === 'éxito') {
          alert(res.mensaje);
          this.cargarMalas();
        } else {
          alert("Error: " + res.error);
        }
        this.cargando = false;
      },
      error: (err) => {
        console.error('Error en la corrección:', err);
        alert("Ocurrió un error al intentar actualizar Pinecone.");
        this.cargando = false;
      }
    });
  }
}