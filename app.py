from flask import Flask, request, jsonify
import numpy as np
import pywt
import tensorflow as tf

app = Flask(__name__)

# Linear Regression Route
@app.route('/linear_regression', methods=['POST'])
def linear_regression():    
    data = request.json
    x = np.array(data['x'])
    y = np.array(data['y'])

    slope, intercept = np.polyfit(x, y, 1)

    response = {'slope': slope, 'intercept': intercept}
    return jsonify(response)

# Denoising Route
@app.route('/denoisedPost', methods=['POST'])
def denoise():
    data = request.json
    signal = np.array(data['signal'])

    denoised_method = data['denoised_method']
    if denoised_method == 'dwt':
        denoised_signal = denoised_signal_dwt(signal)
    elif denoised_method == 'nn':
        denoised_signal = denoised_signal_nn(signal)
    else:
        return jsonify({'error': 'Invalid denoised method'})

    response = {'denoised_signal': denoised_signal.tolist()}
    return jsonify(response)

def denoised_signal_dwt(signal, wavelet='bior3.3', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(signal)))
    coeffs = [pywt.threshold(c, threshold) for c in coeffs]
    denoised_signal = pywt.waverec(coeffs, wavelet)
    return denoised_signal

def build_denoising_nn(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(input_shape[0], activation='linear')
    ])
    return model

def denoised_signal_nn(signal):
    # Ubah dimensi input sinyal menjadi (None, panjang_sinyal)
    signal_reshaped = signal.reshape(-1, len(signal))

    model = build_denoising_nn(input_shape=(len(signal),))
    model.compile(optimizer='adam', loss='mean_squared_error')

    noisy_signal = signal_reshaped + np.random.normal(0, 0.1, signal_reshaped.shape)
    model.fit(noisy_signal, signal_reshaped, epochs=50, verbose=0)
    denoised_signal = model.predict(noisy_signal)
    return denoised_signal.flatten()

if __name__ == '__main__':
    app.run(debug=True)
