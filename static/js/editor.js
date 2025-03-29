// Initialize Quill editor with the transcript content
var quill = new Quill('#editor-container', {
    theme: 'snow'
  });
  quill.root.innerHTML = transcriptContent;
  
  // Update hidden input before form submission
  function updateEditorContent() {
    document.getElementById("html_content").value = quill.root.innerHTML;
  }
  
  // Modal handling
  var renameModal = document.getElementById("renameModal");
  var openRenameModalBtn = document.getElementById("openRenameModal");
  var closeRenameModalBtn = document.getElementById("closeRenameModal");
  var applyRenameBtn = document.getElementById("applyRenameBtn");
  
  // Open the modal
  openRenameModalBtn.onclick = function() {
    renameModal.style.display = "block";
  };
  
  // Close the modal
  closeRenameModalBtn.onclick = function() {
    renameModal.style.display = "none";
  };
  
  // Apply renaming and close
  applyRenameBtn.onclick = function() {
    applyRename(); // Provided by rename.js
    renameModal.style.display = "none";
  };
  
  // Close modal if user clicks outside the modal content
  window.onclick = function(event) {
    if (event.target === renameModal) {
      renameModal.style.display = "none";
    }
  };
  