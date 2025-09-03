document.addEventListener('DOMContentLoaded', () => {
  const dz = document.querySelector('.dropzone');
  const input = document.querySelector('input[type="file"][name="file"]');
  const preview = document.querySelector('.preview');
  if (!dz || !input) return;

  const showPreview = (file) => {
    if (!preview) return;
    const img = document.createElement('img');
    img.className = 'img-fluid';
    img.alt = 'preview';
    const reader = new FileReader();
    reader.onload = e => { img.src = e.target.result; };
    reader.readAsDataURL(file);
    preview.innerHTML = '';
    preview.appendChild(img);
  };

  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
  dz.addEventListener('drop', e => {
    e.preventDefault();
    dz.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      input.files = e.dataTransfer.files;
      showPreview(e.dataTransfer.files[0]);
    }
  });

  input.addEventListener('change', e => {
    const file = e.target.files && e.target.files[0];
    if (file) showPreview(file);
  });
});

$(document).ready( function() {
    $(document).on('change', '.btn-file :file', function() {
    var input = $(this),
        label = input.val().replace(/\\/g, '/').replace(/.*\//, '');
    input.trigger('fileselect', [label]);
    });

    $('.btn-file :file').on('fileselect', function(event, label) {
        
        var input = $(this).parents('.input-group').find(':text'),
            log = label;
        
        if( input.length ) {
            input.val(log);
        } else {
            if( log ) alert(log);
        }
    
    });
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            
            reader.onload = function (e) {
                $('#img-upload').attr('src', e.target.result);
            }
            
            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#imgInp").change(function(){
        readURL(this);
    }); 	
});