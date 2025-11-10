document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    const loadingSection = document.getElementById('loadingSection');
    const resultsSection = document.getElementById('resultsSection');
    const errorSection = document.getElementById('errorSection');
    const grayImage = document.getElementById('grayImage');
    const colorizedImage = document.getElementById('colorizedImage');
    const errorText = document.getElementById('errorText');

    // File input change event
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // Drag and drop functionality
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    // Click to upload
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });

    function handleFileUpload(file) {
        // Validate file type
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
        if (!allowedTypes.includes(file.type)) {
            showError('Please select a valid image file (JPEG, PNG, GIF, or BMP)');
            return;
        }

        // Validate file size (16MB max)
        const maxSize = 16 * 1024 * 1024;
        if (file.size > maxSize) {
            showError('File size must be less than 16MB');
            return;
        }

        // Show loading
        showLoading();

        // Create FormData and upload
        const formData = new FormData();
        formData.append('file', file);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            
            if (data.success) {
                showResults(data.gray_image, data.colorized_image);
            } else {
                showError(data.error || 'An error occurred while processing the image');
            }
        })
        .catch(error => {
            hideLoading();
            showError('Network error: ' + error.message);
        });
    }

    function showLoading() {
        hideAllSections();
        loadingSection.style.display = 'block';
    }

    function hideLoading() {
        loadingSection.style.display = 'none';
    }

    function showResults(grayImageData, colorizedImageData) {
        hideAllSections();
        
        grayImage.src = grayImageData;
        colorizedImage.src = colorizedImageData;
        
        // Store colorized image data for download
        window.colorizedImageData = colorizedImageData;
        
        resultsSection.style.display = 'block';
        
        // Smooth scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    function showError(message) {
        hideAllSections();
        errorText.textContent = message;
        errorSection.style.display = 'block';
    }

    function hideAllSections() {
        loadingSection.style.display = 'none';
        resultsSection.style.display = 'none';
        errorSection.style.display = 'none';
    }

    // Global functions for buttons
    window.downloadImage = function() {
        if (window.colorizedImageData) {
            const link = document.createElement('a');
            link.href = window.colorizedImageData;
            link.download = 'colorized_image.png';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    };

    window.resetUpload = function() {
        hideAllSections();
        fileInput.value = '';
        window.colorizedImageData = null;
        
        // Smooth scroll to upload area
        uploadArea.scrollIntoView({ behavior: 'smooth' });
    };

    // Add some visual feedback for image loading
    grayImage.addEventListener('load', function() {
        this.style.opacity = '0';
        this.style.transform = 'scale(0.8)';
        setTimeout(() => {
            this.style.transition = 'all 0.5s ease';
            this.style.opacity = '1';
            this.style.transform = 'scale(1)';
        }, 100);
    });

    colorizedImage.addEventListener('load', function() {
        this.style.opacity = '0';
        this.style.transform = 'scale(0.8)';
        setTimeout(() => {
            this.style.transition = 'all 0.5s ease';
            this.style.opacity = '1';
            this.style.transform = 'scale(1)';
        }, 200);
    });
});