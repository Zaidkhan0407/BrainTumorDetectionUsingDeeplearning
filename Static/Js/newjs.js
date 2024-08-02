$(document).ready(function () {
    // Function to show selected image
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').attr('src', e.target.result);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }

    // Triggered when file input changes
    $("#imageUpload").change(function () {
        readURL(this);
        $('.image-section').fadeIn(600); // Fade in the image section
    });

    // Triggered when predict button is clicked
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').removeClass('hidden').addClass('visible');
                if (data.result === 'Not an MRI Image') {
                    $('#result').css('color', 'red');
                } else {
                    $('#result').css('color', 'green');
                }
                $('#result').text('Result: ' + data.result + ', Symmetry Score: ' + data.symmetry_score);
                console.log('Success!');
            },
        });
    });
});