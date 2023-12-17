# ES2022 - 實作9: AI起手式之圖像分類實作

## Lab 9-1 AI應用起手式: 使用 TensorFlow Hub 進行圖像分類, 1W

![image](https://github.com/ba1213022/ES-Fall2023/assets/145248354/19d91c2e-db7f-4f28-88f7-efc0c563f2dc)


## Lab 9-2 實作練習 1W

### 實作1: 從已提供的選項中,找1張自己喜歡的照片來試試看

![image](https://user-images.githubusercontent.com/89304181/202887347-b342b643-bfc0-41d0-9ba0-9e47377776b5.png)

### 實作2: 從網路上找3張自己喜歡的照片來試試看 (jpg/png)

![image](https://github.com/ba1213022/ES-Fall2023/assets/145248354/85492f77-47cb-419e-9397-43859d75f1b9)

![image](https://github.com/ba1213022/ES-Fall2023/assets/145248354/8f423b5f-1d71-4f9e-877d-5f14f1d17747)

![image](https://github.com/ba1213022/ES-Fall2023/assets/145248354/636e741a-9a20-44f0-b9ab-f728d6b6c89d)


### Reference Code
```python
image_name = 'basketball'

images_for_test_map = {
    "tiger": "https://upload.wikimedia.org/wikipedia/commons/b/b0/Bengal_tiger_%28Panthera_tigris_tigris%29_female_3_crop.jpg",
    "bus": "https://upload.wikimedia.org/wikipedia/commons/6/63/LT_471_%28LTZ_1471%29_Arriva_London_New_Routemaster_%2819522859218%29.jpg",
    "car": "https://upload.wikimedia.org/wikipedia/commons/4/49/2013-2016_Toyota_Corolla_%28ZRE172R%29_SX_sedan_%282018-09-17%29_01.jpg",
    "cat": "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg",
    "dog": "https://upload.wikimedia.org/wikipedia/commons/archive/a/a9/20090914031557%21Saluki_dog_breed.jpg",
    "apple": "https://upload.wikimedia.org/wikipedia/commons/1/15/Red_Apple.jpg",
    "turtle": "https://upload.wikimedia.org/wikipedia/commons/8/80/Turtle_golfina_escobilla_oaxaca_mexico_claudio_giovenzana_2010.jpg",
    "flamingo": "https://upload.wikimedia.org/wikipedia/commons/b/b8/James_Flamingos_MC.jpg",
    "piano": "https://upload.wikimedia.org/wikipedia/commons/d/da/Steinway_%26_Sons_upright_piano%2C_model_K-132%2C_manufactured_at_Steinway%27s_factory_in_Hamburg%2C_Germany.png",
    "honeycomb": "https://upload.wikimedia.org/wikipedia/commons/f/f7/Honey_comb.jpg",
    "teapot": "https://upload.wikimedia.org/wikipedia/commons/4/44/Black_tea_pot_cropped.jpg",
    "basketball":"https://www.wilson.com/en-us/media/catalog/product/article_images/WTB7500ID_/WTB7500ID__b722ae318490e0f2e686864dc70fd730.png",
    "f1":"https://photo.8891.com.tw/nc/newcar/article/2023/02/16/1676544008190661_1400_0.jpg",
    "ducati":"https://www.insidemotorcycles.com/wp-content/uploads/2023/03/JRB9358_UC494245_High-resized.jpg"
}

img_url = images_for_test_map[image_name]
image, original_image = load_image(img_url, image_size, dynamic_size, max_dynamic_size)
#show_image(image, 'Experiment image')

#%time 

probabilities = tf.nn.softmax(classifier(image)).numpy()

top_5 = tf.argsort(probabilities, axis=-1, direction="DESCENDING")[0][:5].numpy()
np_classes = np.array(classes)
includes_background_class = probabilities.shape[1] == 1001
print(len(top_5), top_5)


for i, item in enumerate(top_5):
  try:   
    class_index = item if not includes_background_class else item - 1
    
    line = f'({i+1}) {class_index:4} - {classes[class_index]}: {probabilities[0][top_5][i]}'
    animal = classes[class_index]
    translation1 = classes[class_index] #E_2_TW.translate(classes[class_index])
    print(line, ', ', translation1)
    #print(line, ', ', E_2_TW.translate(animal),',', translation1)
  except:
    print("Something Wrong in Translation")

    
show_image(image, '')

### Final Result, 2022.11.19
from datetime import datetime
today = datetime.now()
print('*** Done by %s at ' % ts3,today, type(today))

```

