let typingTimeout = null;
let typingId = 0;
function setCustomText(newText) {
    const textElement = document.getElementById("milo_text");
    textElement.textContent = "";
    typingId++;
    const currentId = typingId;

    let index = 0;
    const speed = 50;

    function typeLetter() {

        if (currentId !== typingId) return;

        if (index < newText.length) {
            textElement.textContent += newText.charAt(index);
            index++;
            typingTimeout = setTimeout(typeLetter, speed);
        }
    }

    if (typingTimeout) {
        clearTimeout(typingTimeout);
        typingTimeout = null;
    }

    typeLetter();
}
