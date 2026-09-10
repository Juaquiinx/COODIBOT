import { Component } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';

@Component({
  selector: 'app-contacto',
  standalone: false,
  templateUrl: './contacto.component.html',
  styleUrl: './contacto.component.css'
})
export class ContactoComponent {
 contactForm: FormGroup;
  mensajeEnviado = false;

  constructor(private fb: FormBuilder) {
    this.contactForm = this.fb.group({
      nombre: ['', Validators.required],
      correo: ['', [Validators.required, Validators.email]],
      mensaje: ['', Validators.required]
    });
  }

  enviarFormulario() {
    if (this.contactForm.valid) {
      const datos = this.contactForm.value;
      console.log('Mensaje enviado:', datos);
      // Aquí puedes llamar a tu servicio de backend para enviar el mensaje
      this.mensajeEnviado = true;
      this.contactForm.reset();
    }
  }
}
