function applyRename(){
    // Get the current HTML content from the Quill editor.
    var htmlContent = quill.root.innerHTML;
    var inputs = document.querySelectorAll('input[type="text"]');
    inputs.forEach(function(input) {
        var oldName = input.name;
        var newName = input.value;
        if(newName){
            var regex = new RegExp(oldName, "g");
            htmlContent = htmlContent.replace(regex, newName);
        }
    });
    // Update the Quill editor with the renamed content.
    quill.root.innerHTML = htmlContent;
}
