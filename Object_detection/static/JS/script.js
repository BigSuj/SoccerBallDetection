function showUploadConfirmation(input) {
  if (input.files && input.files[0]) {
    const file = input.files[0];
    const fileType = file['type'];
    const validImageTypes = ['image/gif', 'image/jpeg', 'image/png', 'image/webp'];
    
    if (!validImageTypes.includes(fileType)) {
      const uploadConfirmation = document.getElementById("uploadConfirmation");
      uploadConfirmation.innerHTML = "";
      uploadConfirmation.style.display = "none";
      
      const preview = document.getElementById("preview");
      preview.src = "";
      preview.style.display = "none";
      
      const wrapper = preview.parentElement;
      wrapper.classList.remove('has-image');
      
      const alert = document.createElement("div");
      alert.className = "alert";
      alert.innerText = "Invalid file type. Please choose a valid image file.";
      
      const container = document.querySelector(".image-preview-container");
      container.appendChild(alert);
      
      setTimeout(function() {
        container.removeChild(alert);
      }, 3000);
      
      input.value = null;
      
      return false;
    }
    
    const uploadConfirmation = document.getElementById("uploadConfirmation");
    uploadConfirmation.innerHTML = "Image uploaded:";
    uploadConfirmation.style.display = "block";
    
    const preview = document.getElementById("preview");
    const wrapper = preview.parentElement;
    wrapper.classList.add('has-image');
    
    const reader = new FileReader();
    reader.onload = function(e) {
      preview.src = e.target.result;
      preview.style.display = "block";
    }
    reader.readAsDataURL(input.files[0]);
  } else {
    const uploadConfirmation = document.getElementById("uploadConfirmation");
    uploadConfirmation.innerHTML = "";
    uploadConfirmation.style.display = "none";
    
    const preview = document.getElementById("preview");
    preview.src = "";
    preview.style.display = "none";
    
    const wrapper = preview.parentElement;
    wrapper.classList.remove('has-image');
  }
}

const predictBtn = document.getElementById("predict-btn");
predictBtn.addEventListener("click", function(event) {
  // Prevent the form from submitting if no image has been uploaded
  if (!document.querySelector(".has-image")) {
    event.preventDefault();
    alert("Please upload an image before predicting.");
  }
});


