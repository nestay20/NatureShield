# Plants
curl -L `
  "https://tfhub.dev/google/lite-model/aiy/vision/classifier/plants_V1/3?tf-hub-format=compressed" `
  -o plants_v1.tar.gz

tar -xzf plants_v1.tar.gz -C .

# Birds
curl -L `
  "https://tfhub.dev/google/lite-model/aiy/vision/classifier/birds_V1/3?tf-hub-format=compressed" `
  -o birds_v1.tar.gz

tar -xzf birds_v1.tar.gz -C .

# Insects
curl -L `
  "https://tfhub.dev/google/lite-model/aiy/vision/classifier/insects_V1/3?tf-hub-format=compressed" `
  -o insects_v1.tar.gz

tar -xzf insects_v1.tar.gz -C .
