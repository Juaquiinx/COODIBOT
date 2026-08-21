document.addEventListener('DOMContentLoaded', () => {
    const coodibotBtn = document.getElementById('coodibotBtn');
    const chatWindow = document.getElementById('chatWindow');
    const closeChatBtn = document.getElementById('closeChatBtn');
    const sendMessageBtn = document.getElementById('sendMessageBtn');
    const userInput = document.getElementById('userInput');
    const chatMessages = document.getElementById('chatMessages');

    // Generamos un ID de sesión simple para la base de datos SQLite
    const sessionID = "sesion_docente_" + Math.floor(Math.random() * 1000);

    // Abrir/Cerrar la ventana de chat al hacer clic en el widget
    coodibotBtn.addEventListener('click', () => {
        chatWindow.classList.toggle('hidden');
    });

    // Cerrar desde la "X"
    closeChatBtn.addEventListener('click', () => {
        chatWindow.classList.add('hidden');
    });

    // Enviar mensaje al hacer clic en el botón
    sendMessageBtn.addEventListener('click', enviarMensaje);

    // Enviar mensaje al presionar "Enter"
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            enviarMensaje();
        }
    });

    async function enviarMensaje() {
        const texto = userInput.value.trim();
        if (texto === '') return;

        // 1. Dibujar el mensaje del usuario en la pantalla
        agregarMensaje(texto, 'user-message');
        userInput.value = '';

        // 2. Mostrar indicador de "escribiendo..." (Opcional pero recomendado)
        const escribiendoId = agregarMensaje('COODIBOT está pensando...', 'bot-message');

        try {
            // 3. Conexión real con tu Backend FastAPI (main.py)
            const respuesta = await fetch('http://127.0.0.1:8000/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    pregunta: texto,
                    session_id: sessionID
                })
            });

            const datos = await respuesta.json();

            // 4. Remover el "escribiendo..." y mostrar la respuesta real
            document.getElementById(escribiendoId).remove();

            if (datos.respuesta) {
                // Reemplazamos los saltos de línea (\n) por <br> para HTML
                const respuestaFormateada = datos.respuesta.replace(/\n/g, '<br>');
                agregarMensaje(respuestaFormateada, 'bot-message');
            } else {
                agregarMensaje("Hubo un error al procesar tu solicitud.", 'bot-message');
            }

        } catch (error) {
            document.getElementById(escribiendoId).remove();
            agregarMensaje("Error de conexión con el servidor.", 'bot-message');
            console.error("Error:", error);
        }
    }

    // Función auxiliar para inyectar los globos de chat en el HTML
    function agregarMensaje(texto, claseCSS) {
        const div = document.createElement('div');
        div.className = `message ${claseCSS}`;
        div.innerHTML = texto;

        // Asignamos un ID temporal por si necesitamos borrarlo (ej. "escribiendo...")
        const idTemp = 'msg-' + Date.now();
        div.id = idTemp;

        chatMessages.appendChild(div);

        // Auto-scroll hacia el último mensaje
        chatMessages.scrollTop = chatMessages.scrollHeight;

        return idTemp;
    }
});