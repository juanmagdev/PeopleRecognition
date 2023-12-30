import React, { useState, useEffect } from 'react';
import axios from 'axios';

const App = () => {
  const [image, setImage] = useState(null);

  useEffect(() => {
    const fetchLatestImage = async () => {
      try {
        const response = await axios.get('http://localhost:3001/api/latest-image').
        then(response => {
          setImage(response.data.contenido);}).
        catch((err) => console.log(err));
        console.log(response);

        // setImage(response.data.image);
      } catch (error) {
        console.error('Error al obtener la imagen:', error);
      }
    };

    //tiempo de actualizacion de la imagen
    const intervalId = setInterval(fetchLatestImage, 1000);

    return () => clearInterval(intervalId);
  }, []);

  return (
    <>
      <h1>Camara de Seguridad</h1>
      {image && <img src={`data:image/png;base64, ${image}`} alt="Imagen de la cámara" />}
      <p>La imagen se actualiza cada 5 segundos</p>
    </>
  );
};

export default App;
