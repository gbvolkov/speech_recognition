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

