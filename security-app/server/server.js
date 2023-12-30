const express = require('express');
const multer = require('multer');
const app = express();
const cors = require('cors');
const port = 3001;
const fs = require('fs');

const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

app.use(cors());

app.post('/api/upload', upload.single('image'), (req, res) => {
  lastImage = req.file.buffer;
  res.send('Imagen recibida con éxito');
  base64Image = req.file.buffer.toString('base64');

  fs.writeFile('image.txt', base64Image, { encoding: 'base64' }, function(err) {

  });
  

});

let lastImage = null;

app.get('/api/latest-image', (req, res) => {
  // configuro headers para que el navegador entienda que le estoy enviando una imagen
  if (lastImage) {
    const base64Image = lastImage.toString('base64');
    const contenido = fs.readFileSync('image.txt', 'utf-8');
    console.log(contenido);
    res.json({ contenido });

  } else {
    res.status(404).send('No hay imagen disponible');
  }
});

app.listen(port, () => {
  console.log(`Servidor escuchando en http://localhost:${port}`);
});