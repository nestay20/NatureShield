// Natureshield App.js

// Polyfill Buffer for React Native JS environment
import { Buffer } from 'buffer';
if (typeof global.Buffer === 'undefined') {
  global.Buffer = Buffer;                   
}

import React, { useState, useEffect, useRef } from 'react';
// React native imports for displaying text and things 
import {
  SafeAreaView,                           
  View,                                   
  Text,                                  
  Button,                                 
  Image,                                  
  TouchableOpacity,                       
  Platform,                               
} from 'react-native';

import { launchImageLibrary } from 'react-native-image-picker';  // For image picker
import Tflite from 'tflite-react-native';                        // Tensorflow for react native
import RNFS from 'react-native-fs';                              // To access files on device
import { parse } from 'csv-parse/sync';                          // To parse labels for models

// The size that our models expect for input images
const INPUT_SIZE = 224;

// Get the float32 models and their labels
const MODEL_CONFIG = {
  Plants:  { model: 'models/plants/model_float32.tflite',  labels: 'models/plants/labels.csv' },
  Birds:   { model: 'models/birds/model_float32.tflite',   labels: 'models/birds/labels.csv' },
  Insects: { model: 'models/insects/model_float32.tflite', labels: 'models/insects/labels.csv' },
  Mammals: { model: 'models/mammals/model_float32.tflite', labels: 'models/mammals/labels.csv' },
};

export default function App() {
  // Track which category is selected (for testing)
  const [category, setCategory] = useState('Plants');

  // Map the ID to name
  const [labelsMap, setLabelsMap] = useState({});

  // URI of the last image
  const [uri, setUri] = useState(null);

  // Textual result to display
  const [result, setResult] = useState('');

  // Reference our tensorflow lite interpreterd
  const tflite = useRef(new Tflite()).current;

  // Reload the corresponding CSV when we switch categories
  useEffect(() => {
    async function loadCSV() {
      try {
        const { labels: labelsPath } = MODEL_CONFIG[category];

        // Read the CSV
        const text = Platform.OS === 'android'
          ? await RNFS.readFileAssets(labelsPath, 'utf8')
          : await RNFS.readFile(`${RNFS.MainBundlePath}/${labelsPath}`, 'utf8');

        // Parse t he CSV
        const recs = parse(text, { skip_empty_lines: true });

        // Builds the map, will come in handy when we add common and scientific names
        const map = {};
        recs.forEach(row => {
          const id = parseInt(row[0], 10);
          if (!isNaN(id)) {
            const sci = row[1] || '';
            const com = row[2] || '';
            map[id] = com ? `${com} (${sci})` : sci;
          }
        });

        setLabelsMap(map);
        // Error catch
      } catch (e) {
        console.error('CSV load error', e);
        setLabelsMap({});
      }
    }

    loadCSV();
  }, [category]);

  // Reload the corresponding model.tflite when we switch categories
  useEffect(() => {
    const { model, labels } = MODEL_CONFIG[category];

    tflite.loadModel(
      {
        model,            // path to the .tflite model
        labels,           // path to the .csv labels
        isQuantized: false, // Ensure quantized is false
        numThreads: 1,    // For safety ensure we only use 1 thread
      },
      err => {
        if (err) console.error('loadModel error', err);
      }
    );

    // Clean up interpreter when we switch categories
    return () => tflite.close();
  }, [category]);

  // Choose an image and then run tflite to infer from that image
  const Classify = () => {
    launchImageLibrary({ mediaType: 'photo' }, resp => {
      if (resp.didCancel || resp.errorCode) return;

      const picked = resp.assets[0].uri;
      setUri(picked);
      
      // Run tflite model on image
      tflite.runModelOnImage(
        {
          path: picked,     // local URI of the image
          imageMean: 0.0,   // for float32, mean=0
          imageStd: 255.0,  // scale pixel values [0–255] to [0–1]
          numResults: 1,    // we only want top result
          threshold: 0.01,  // confidence threshold
          width: INPUT_SIZE,
          height: INPUT_SIZE,
        },
        (err, res) => {
          // Check for errors and unforeseen events
          if (err) {
            console.error('Inference error', err);
            setResult('Error');
          } else if (res && res.length) {
            const idx = res[0].index;               // Model returns 0, shouldn't happen but can happen
            setResult(labelsMap[idx] ?? 'Unknown'); // Lookup the index # if it doesn't exist then species is "Unknown"
          } else {
            setResult('No result');
          }
        }
      );
    });
  };






  // Everything below is for the testing UI, if we are not able to implement this into our react native UI in time we could use this for now
  return (
    <SafeAreaView style={{ flex: 1, padding: 20 }}>
      {/* Category Picker */}
      <View style={{ flexDirection: 'row', marginBottom: 20 }}>
        {Object.keys(MODEL_CONFIG).map(cat => (
          <TouchableOpacity
            key={cat}
            onPress={() => {
              setCategory(cat);   // switch model
              setResult('—');     // reset result display
              setUri(null);       // clear image preview
            }}
            style={{
              padding: 10,
              backgroundColor: category === cat ? '#888' : '#ccc',
              marginRight: 8,
            }}
          >
            <Text>{cat}</Text>
          </TouchableOpacity>
        ))}
      </View>

      <Button title="Upload Image" onPress={Classify} />

      {/* Image Preview */}
      {uri && (
        <Image
          source={{ uri }}
          style={{ width: INPUT_SIZE, height: INPUT_SIZE, marginTop: 20 }}
        />
      )}
      {/* Display Results */}
      <Text style={{ fontSize: 16, marginTop: 20 }}>Result: {result}</Text>
    </SafeAreaView>
  );
}
