import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { NavbarComponent } from './components/navbar/navbar.component';
import { HomeComponent } from './components/home/home.component';
import { FooterComponent } from './components/footer/footer.component';
import { AprendeComponent } from './components/educacion/aprende/aprende.component';
import { QuienesComponent } from './components/educacion/quienes/quienes.component';
import { ContactoComponent } from './components/contacto/contacto.component';
import { ReactiveFormsModule } from '@angular/forms';
import { ServiciosComponent } from './components/servicios/servicios.component';
import { PageNotFoundComponent } from './components/page-not-found/page-not-found.component';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import { CoodibotChatComponent } from './components/coodibot-chat/coodibot-chat.component';
import { AdminComponent } from './components/admin/admin.component';
@NgModule({
  declarations: [
    AppComponent,
    NavbarComponent,
    HomeComponent,
    FooterComponent,
    AprendeComponent,
    QuienesComponent,
    ContactoComponent,
    ServiciosComponent,
    PageNotFoundComponent,
    CoodibotChatComponent,
    AdminComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    ReactiveFormsModule,
    FormsModule,
    HttpClientModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
