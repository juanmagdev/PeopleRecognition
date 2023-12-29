const express = require('express');
const multer = require('multer');
const app = express();
const cors = require('cors');
const port = 3001;

const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

app.use(cors());

app.post('/api/upload', upload.single('image'), (req, res) => {
  // lastImage = req.file.buffer;
  // console.log(req.file.buffer);
  lastImage = req.file.buffer;
  res.send('Imagen recibida con éxito');
});

let lastImage = null;

app.get('/api/latest-image', (req, res) => {
  console.log(lastImage);
  if (lastImage) {
    const base64Image = lastImage.toString('base64');
    res.send({ image: base64Image });
    // res.send({ image: 'iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg==' });
    res.send({ image: lastImage });
  } else {
    res.status(404).send('No hay imagen disponible');
  }
});

app.listen(port, () => {
  console.log(`Servidor escuchando en http://localhost:${port}`);
});


