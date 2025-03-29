function applyRename() {
    let inputs = document.querySelectorAll('input[type="text"]');
    inputs.forEach(input => {
        let oldName = input.name;
        let newName = input.value.trim();
        if (newName) {
            // Replace only text nodes containing oldName
            document.querySelectorAll("#editor-container *").forEach(node => {
                node.childNodes.forEach(child => {
                    if (child.nodeType === Node.TEXT_NODE && child.nodeValue.includes(oldName)) {
                        child.nodeValue = child.nodeValue.replaceAll(oldName, newName);
                    }
                });
            });
        }
    });
}