document.addEventListener('DOMContentLoaded', function () {
    const imageSelector = document.getElementById('imageSelector');// 获取下拉框和输出容器的引用
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const modeSelector = document.getElementById('modeSelector');  // 获取下拉框和输出容器的引用
    const image = document.getElementById('image');
    const imageContainer = document.querySelector('.image-container'); // 在 image-container 中创建一个新的 div 来展示画框
    const selectionBox = document.createElement('div');
    let startPoint = null;   // 用于存储画框模式下的起始点和终点
    let endPoint = null;
    let currentMode = modeSelector.value;  // 用于存储当前模式，默认为单击
    let isMouseDown = false;
    selectionBox.setAttribute('id', 'selectionBox');
    selectionBox.style.position = 'absolute';
    selectionBox.style.border = '2px solid #00ff00';
    selectionBox.style.display = 'none';
    imageContainer.appendChild(selectionBox);

    // 监听下拉框的改变事件
    modeSelector.addEventListener('change', function (event) {
        clearPreviousSelectionBox()
        clearMask()
        currentMode = event.target.value;
        // 重置起始点和终点
        startPoint = null;
        endPoint = null;
    });

    // 鼠标按下事件
    image.addEventListener('mousedown', function (event) {
        if (currentMode === 'Box') {
            clearPreviousSelectionBox()
            const rect = image.getBoundingClientRect();
            isMouseDown = true;
            startPoint = {
                x: event.clientX,
                y: event.clientY
            };
            selectionBox.style.left = `${startPoint.x}px`;
            selectionBox.style.top = `${startPoint.y}px`;
            selectionBox.style.width = '0px';
            selectionBox.style.height = '0px';
            selectionBox.style.display = 'block';
            event.preventDefault();
        }
    });
    // 鼠标移动，实时计算坐标，并画框
    image.addEventListener('mousemove', function (event) {
        if (isMouseDown && currentMode === 'Box') {
            const rect = image.getBoundingClientRect();
            const currentPoint = {
                x: event.clientX,
                y: event.clientY
            };
            console.log(event.offsetX, event.offsetY, event.clientX, event.clientY, rect.left, rect.top)
            const width = currentPoint.x - startPoint.x;
            const height = currentPoint.y - startPoint.y;
            selectionBox.style.width = `${Math.abs(width)}px`;
            selectionBox.style.height = `${Math.abs(height)}px`;
            // 使用 Math.min 确保 left 和 top 值是起始点和当前点中的较小值
            selectionBox.style.left = `${Math.min(startPoint.x, currentPoint.x)}px`;
            selectionBox.style.top = `${Math.min(startPoint.y, currentPoint.y)}px`;
            event.preventDefault();
        }
    });

    // 鼠标释放事件
    imageContainer.addEventListener('mouseup', function (event) {
        if (isMouseDown && currentMode === 'Box') {
            isMouseDown = false;
            const endPoint = { x: event.clientX, y: event.clientY };
            // 发送坐标到后端
            sendBoxCoordinates(startPoint, endPoint);
            const width = endPoint.x - startPoint.x;
            const height = endPoint.y - startPoint.y;
            selectionBox.style.width = `${Math.abs(width)}px`;
            selectionBox.style.height = `${Math.abs(height)}px`;
            // 使用 Math.min 确保 left 和 top 值是起始点和当前点中的较小值
            selectionBox.style.left = `${Math.min(startPoint.x, endPoint.x)}px`;
            selectionBox.style.top = `${Math.min(startPoint.y, endPoint.y)}px`;
            event.preventDefault();

        }
    });

    // 发送坐标到后端的函数
    function sendBoxCoordinates(startPoint, endPoint) {
        // 计算实际图片上的坐标
        const rect = image.getBoundingClientRect();
        const boxStart = {
            x: Math.round(startPoint.x - rect.left),
            y: Math.round(startPoint.y - rect.top)
        }
        const boxEnd = {
            x: Math.round(endPoint.x - rect.left),
            y: Math.round(endPoint.y - rect.top)
        };
        // 发送坐标
        fetch('http://localhost:8000/box-coordinates', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ start: boxStart, end: boxEnd }),
        })
            .then(response => response.json())
            .then(maskData => {
                console.log('Backend response:', maskData);
                displayMask(maskData.maskList);
            })
            .catch(error => {
                console.error('Error sending box coordinates:', error);
            });
    }

    // 清除之前的框
    function clearPreviousSelectionBox() {
        selectionBox.style.width = '0px';
        selectionBox.style.height = '0px';
        selectionBox.style.display = 'none';
    }

    // 清除之前的mask
    function clearMask() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    // 获取图片列表并填充下拉框选项
    function fetchImageList() {
        fetch('http://localhost:8000/images', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            }
        })
            .then(response => response.json())
            .then(images => {
                images.forEach(image => {
                    const option = document.createElement('option');
                    option.value = image;
                    option.textContent = image;
                    imageSelector.appendChild(option);
                });
            })
            .catch(error => {
                console.error('Error fetching image list:', error);
            });
    }

    // 发送所选图片名称到后端并启动程序
    function sendSelectedImage(imageName) {
        fetch('http://localhost:8000/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ imageName: imageName }),
        })
            .then(response => response.json())
            .then(data => {
                console.log('Start command response:', data);
            })
            .catch(error => {
                console.error('Error sending selected image:', error);
            });
    }

    // 计算鼠标的位置，并发送给后端
    function sendCoordinates(event) {
        if (currentMode === 'Point') {
            clearPreviousSelectionBox()
            const rect = image.getBoundingClientRect();
            const mouseX = event.clientX - rect.left;
            const mouseY = event.clientY - rect.top;

            // Assuming h and l are the resolution of the image
            const h = image.naturalHeight;
            const w = image.naturalWidth;

            // Create an object with coordinates
            const coordinates = {
                x: mouseX,
                y: mouseY,
                w: w,
                h: h

            };
            // Send coordinates to the backend
            sendToBackend(coordinates);
        }
    }

    // Function to send coordinates to the backend
    function sendToBackend(coordinates) {
        const apiUrl = 'http://localhost:8000/coordinates/1';
        // Assuming you are using fetch API
        fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(coordinates),
        })
            .then(response => response.json())
            .then(maskData => {
                console.log('Backend response:', maskData);
                displayMask(maskData.maskList);
            })
            .catch(error => {
                console.error('Error sending coordinates:', error);
            });
    }

    // Function to display the mask over the image
    function displayMask(maskData) {
        // 填充 imageData
        const imageData = ctx.createImageData(canvas.width, canvas.height);
        for (let y = 0; y < canvas.height; y++) {
            for (let x = 0; x < canvas.width; x++) {
                const index = (y * canvas.width + x) * 4;
                imageData.data[index] = 0; // R value
                imageData.data[index + 1] = 0; // G value
                imageData.data[index + 2] = 255; // B value
                // 根据后端返回的数据格式，这里可能需要调整
                imageData.data[index + 3] = maskData[y][x] === true ? 255 : 0; // A value
            }
        }
        // 将 imageData 绘制到 canvas 上
        ctx.putImageData(imageData, 0, 0);
    }

    // 监听下拉框的变化
    imageSelector.addEventListener('change', function () {
        clearPreviousSelectionBox()
        clearMask()
        const selectedImageName = this.value;
        if (selectedImageName) {
            // 清除 canvas 内容
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            // 更新图片的 src 属性
            image.src = 'images/' + selectedImageName;
            // 由于图片源可能更改，需要重新计算并设置 canvas 的位置
            image.onload = function () {
                const image = document.getElementById('image');
                canvas.width = image.naturalWidth;
                canvas.height = image.naturalHeight;
                canvas.style.position = 'absolute';
                canvas.style.left = image.offsetLeft + 'px';
                canvas.style.top = image.offsetTop + 'px';
                canvas.style.pointerEvents = 'none'; // 允许点击事件穿透 canvas
                document.body.appendChild(canvas);
            };
            // 发送选中的图片名称到后端
            sendSelectedImage(selectedImageName);
        }
    });

    // Attach click event listener to the image
    image.addEventListener('click', sendCoordinates);
    // 在页面加载时获取图片列表
    fetchImageList();
});
