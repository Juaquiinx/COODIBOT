import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { HomeComponent } from './components/home/home.component';
import { AprendeComponent } from './components/educacion/aprende/aprende.component';
import { QuienesComponent } from './components/educacion/quienes/quienes.component';
import { ContactoComponent } from './components/contacto/contacto.component';
import { ServiciosComponent } from './components/servicios/servicios.component';
import { AdminComponent } from './components/admin/admin.component';
import { PageNotFoundComponent } from './components/page-not-found/page-not-found.component';

const routes: Routes = [
  { path: "", component: HomeComponent },
  { path: "educacion", component: AprendeComponent },
  { path: "quienes", component: QuienesComponent },
  { path: "contacto", component: ContactoComponent },
  { path: "servicios", component: ServiciosComponent },
  { path: "admin", component: AdminComponent },
  { path: '**', component: PageNotFoundComponent }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }