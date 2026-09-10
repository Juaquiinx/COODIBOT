import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CoodibotChatComponent } from './coodibot-chat.component';

describe('CoodibotChatComponent', () => {
  let component: CoodibotChatComponent;
  let fixture: ComponentFixture<CoodibotChatComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [CoodibotChatComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CoodibotChatComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
