import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  standalone: false,
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'COODI: COnecta,Observa,Descompone,Innova';
  showCoodibot = false;

  toggleCoodibot() {
    this.showCoodibot = !this.showCoodibot;
  }

  navigateTo(seccion: string) {
    document.getElementById(seccion)?.scrollIntoView({ behavior: 'smooth' });
  }

}
