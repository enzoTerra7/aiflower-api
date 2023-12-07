from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
from PIL import Image
import io
import base64


app = Flask(__name__)

# Carregar o modelo treinado
MLP = joblib.load("modelo_treinado.joblib")

@app.route('/classify', methods=['POST'])
def classify():
    try:
        print('Received POST request')

        # Obter os dados da imagem do corpo da solicitação POST
        data = request.get_json()

        base64_image = data["image"]

        # Converter a imagem para base64
        image_bytes = base64.b64decode(base64_image)

        # Obter o array numpy da imagem
        image_array = np.array(Image.open(io.BytesIO(image_bytes)))
        
        result = classify_image(image_array, MLP)

        print('return the results')

        print(result)

        # Encontrar a classe com a maior porcentagem
        finalClass = 0
        finalPercent = 0
        className = ''

        for val in result:
          first_number, second_number = val
          if second_number > finalPercent:
            finalClass = first_number
            finalPercent = second_number
          print(val)

        if(finalClass == 0):
            className = 'Margarida'
        if(finalClass == 1):
            className = 'Dente de Leão'
        if(finalClass == 2):
            className = 'Rosa'
        if(finalClass == 3):
            className = 'Girassol'
        if(finalClass == 4):
            className = 'Tulipa'

        # Retornar os resultados
        result = {
            "class_name": className,
            "certainty": finalPercent
        }
        print(f'Results: {result}')

        return jsonify(result)

    except Exception as e:
        print(f'Error: {e}')
        return jsonify({"error": str(e)})

def classify_image(image_64, MLP):
    try:
        # print('load image')
        # # Carregar a imagem
        # image = Image.open(image_path)
        image_array = image_64

        print('get central pixels')
        # Obter os pixels centrais
        height, width, _ = image_array.shape
        central_pixels = image_array[height//4:height*3//4, width//4:width*3//4]

        print('calculate average')
        # Calcular a média dos pixels centrais
        average_red = np.mean(central_pixels[:, :, 0])
        average_green = np.mean(central_pixels[:, :, 1])
        average_blue = np.mean(central_pixels[:, :, 2])

        print('create input data')
        # Criar os dados de entrada
        input_data = np.array([[average_red, average_green, average_blue]])

        print('make prediction')
        # Fazer a previsão
        class_probabilities = MLP.predict_proba(input_data)

        print('get results')
        # Obter o percentual de cada classe
        class_labels = MLP.classes_
        class_percentages = class_probabilities * 100

        print('format results')
        # Formatar os resultados como uma lista de tuplas
        results = [(label, float(percentage)) for label, percentage in zip(class_labels, class_percentages[0])]

        print('return results')
        # Retornar os resultados
        return results

    except Exception as e:
        print('error')
        # Em caso de erro, retornar uma mensagem de erro
        return str(e)

@app.route('/')
def hello_world():
    return 'Hello World!'

CORS(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
