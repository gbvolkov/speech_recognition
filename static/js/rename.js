
function applyRename() {
    let inputs = document.querySelectorAll('input[type="text"]');
    inputs.forEach(input => {
        let oldName = "**"+input.name+"**";
        let newName = input.value.trim();
        if (newName) {
            newName = "**"+newName+"**";
            // Replace only text nodes containing oldName
            document.querySelectorAll("#editor-container *").forEach(node => {
                node.childNodes.forEach(child => {
                    if (child.nodeType === Node.TEXT_NODE && child.nodeValue==oldName) {
                        child.nodeValue = child.nodeValue.replaceAll(oldName, newName);
                    }
                });
            });
        }
    });
}

function updateSpeakerList() {
    // Find all speaker names within the Quill editor's content
    let editorElements = document.querySelectorAll("#editor-container *");
    let speakerElements = Array.from(editorElements).filter(el => /\*\*(.*?)\*\*$/s.test(el.textContent.trim()));
    let speakersSet = new Set();
    speakerElements.forEach(el => {
        let name = el.textContent.trim();
        if (name) {
            speakersSet.add(name.substring(2, name.length - 2));
        }
    });
    // Convert Set to Array for easier processing
    let speakers = Array.from(speakersSet);
    
    // Locate the inner div of the rename form in the sidebar
    let innerDiv = document.querySelector("#renameForm .inner-div");
    if (!innerDiv) return;
    
    // Clear the current list
    innerDiv.innerHTML = "";
    
    // Rebuild the speaker list with updated names
    speakers.forEach(speaker => {
        let label = document.createElement("label");
        label.textContent = speaker + ": ";
        let input = document.createElement("input");
        input.type = "text";
        input.name = speaker;
        input.placeholder = "Enter real name";
        label.appendChild(input);
        innerDiv.appendChild(label);
    });
    setupSpeakerInputEvents();
}

function setupSpeakerInputEvents() {
    const inputs = document.querySelectorAll('#renameForm input[type="text"]');
    inputs.forEach(input => {
      input.addEventListener('focus', function() {
        // When an input is focused, filter the editor for this speaker
        const speakerName = input.name;  // e.g., "Андрей Васильев"
        filterEditorContent(speakerName);
        //alert(speakerName)
      });
      input.addEventListener('blur', function() {
        // Use a small delay to allow focus to shift to another input,
        // otherwise, restore full content if focus left the rename form.
        setTimeout(() => {
          if (!document.querySelector('#renameForm input:focus')) {
            restoreEditorContent();
          }
        }, 100);
      });
    });
  }
  