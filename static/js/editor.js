// Initialize Quill editor with the transcript content
var quill = new Quill('#editor-container', {
    theme: 'snow'
});
quill.root.innerHTML = transcriptContent;
  
// Update hidden input before form submission
function updateEditorContent() {
    document.getElementById("html_content").value = quill.root.innerHTML;
}
  
// Sidebar handling for speaker renaming
var toggleRenameSidebarBtn = document.getElementById("toggleRenameSidebar");
var renameSidebar = document.getElementById("renameSidebar");
var applyRenameBtn = document.getElementById("applyRenameBtn");
  
// Toggle the sidebar on button click
toggleRenameSidebarBtn.onclick = function() {
    renameSidebar.classList.toggle("active");
};
  
// Apply renaming and hide the sidebar after applying changess
applyRenameBtn.onclick = function() {
    applyRename(); // Provided by rename.js
    renameSidebar.classList.remove("active");
    updateSpeakerList();  // Refresh the list of speakers in the sidebar
};

document.addEventListener('DOMContentLoaded', function() {
    // Select all text inputs in the rename form
    const renameInputs = document.querySelectorAll('#renameForm input[type="text"]');
    
    renameInputs.forEach(input => {
      // Prevent keypress if the key is an asterisk
      input.addEventListener('keypress', function(e) {
        if (e.key === '*') {
          e.preventDefault();
        }
      });
      
      // Prevent pasting if the pasted content contains an asterisk
      input.addEventListener('paste', function(e) {
        const pasteData = e.clipboardData.getData('text');
        if (pasteData.includes('*')) {
          e.preventDefault();
        }
      });
    });
    setupSpeakerInputEvents();
  });

  function filterEditorContent(speakerName) {
    const editor = document.querySelector('.ql-editor');
    let currentSpeaker = null;
    
    // Iterate over each paragraph block in order.
    editor.querySelectorAll('p').forEach(block => {
      // Check if this block starts with a speaker header.
      const strongElem = block.querySelector('strong');
      if (strongElem) {
        // Remove markdown asterisks and extra whitespace.
        const nameText = strongElem.textContent.replace(/\*\*/g, '').trim();
        currentSpeaker = nameText;
      }
      // If the current speaker matches the one being filtered, show the block; otherwise hide it.
      if (currentSpeaker === speakerName) {
        block.classList.remove('hidden-block');
      } else {
        block.classList.add('hidden-block');
      }
    });
  }
  
  function restoreEditorContent() {
    const editor = document.querySelector('.ql-editor');
    editor.querySelectorAll('.hidden-block').forEach(block => {
      block.classList.remove('hidden-block');
    });
  }
  