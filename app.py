# pip install flask
# pip install azure-cognitiveservices-vision-computervision
from fileinput import filename
from re import search
from flask import Flask, flash, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw


import os, uuid, sys, requests, random
from io import BytesIO
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.storage.blob import ContentSettings, ContainerClient
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cosmos import exceptions, CosmosClient, PartitionKey
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, QualityForRecognition

UPLOAD_FOLDER = 'static/uploads/'

#================================================================================================================
# DATABASE
#================================================================================================================
# Initialize the Cosmos client
endpoint = "https://db-cosmos-sql-ia2.documents.azure.com:443/"
key = '7kdqzDv9mKfo4iH8hqSPjmVj1X1RBxaF2KwQe7xZlbWIpSvCzfgGhfP5wKi4eAwWhMnAI5oEZkrdAPhXoZ9Wgg=='

# <create_cosmos_client>
client = CosmosClient(endpoint, key)
# </create_cosmos_client>
print("Connexion bdd : ")
print(client)
# Create a database
# <create_database_if_not_exists>
database_name = 'db-cosmos-sql-ia2'
database = client.create_database_if_not_exists(id=database_name)
# </create_database_if_not_exists>

# Create a container
# Using a good partition key improves the performance of database operations.
# <create_container_if_not_exists>
container_name = 'container'
container = database.create_container_if_not_exists(
    id=container_name, 
    partition_key=PartitionKey(path="/lastName"),
    offer_throughput=400
)
# </create_container_if_not_exists>
#================================================================================================================

# Instancier la web app 
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

AZURE_END_POINT='DefaultEndpointsProtocol=https;AccountName=store5jj2uutfdh4us;AccountKey=GqM/P4DPk68WMYUrxoaOFKy0j08o0cuytm8S6xT88XTqKQBs5A5x2dJ4CQ/b+S2+IvPCe6+XCaLS+AStNn5YRw==;EndpointSuffix=core.windows.net'
AZURE_BLOB_NAME='containerblob'

# Initialize the connection to Azure storage account
blob_service_client =  BlobServiceClient.from_connection_string(AZURE_END_POINT)
print("Connexion to blob sucess !")

container_client = ContainerClient.from_connection_string(conn_str=AZURE_END_POINT, container_name=AZURE_BLOB_NAME)

if container_client.exists():
    # Container exists. You can now use it.
    print("blob already exists")
else:
    # Container does not exist. You can now create it.
    blob_service_client.create_container(AZURE_BLOB_NAME, public_access='container')
    print("blob not exists")

# Création de la connexion au computer vision
# cog_key : aller sur le portail azure => Sur le serveur de vision apr ordinateur => Clé et point de terminaisson => Clé 1
cog_key = '4ae65ad8ecef40d9add6f1ad18a38d69'
cog_endpoint = 'https://serv-comp-vision.cognitiveservices.azure.com/'

# Get a client for the computer vision service
computervision_client = ComputerVisionClient(cog_endpoint, CognitiveServicesCredentials(cog_key))

#================================================================================================================
# RECONNAISSANCE FACIALE
#================================================================================================================
# pip install --upgrade azure-cognitiveservices-vision-face
KEY_RECO_FAC='45b2b03ec86c4d929cdf545f6aa60c1c'
ENDPOINT_RECO_FAC='https://reco-facial-ia.cognitiveservices.azure.com/'

# Create an authenticated FaceClient.
face_client = FaceClient(ENDPOINT_RECO_FAC, CognitiveServicesCredentials(KEY_RECO_FAC))
print('Connexion réussi à l api de reconnaissance facial !')
#==============================================================
def recoFacial(filename):
    # Detect a face in an image that contains a single face
    image1 = 'https://store5jj2uutfdh4us.blob.core.windows.net/container-identification/Donald-Trump.jpg'
    # We use detection model 3 to get better performance.
    detected_faces1 = face_client.face.detect_with_url(url=image1, detection_model='detection_03')

    # Detect a face in an image that contains a single face
    # BufferedReader
    image2 = open(filename, 'rb') 

    # We use detection model 3 to get better performance.
    detected_faces2 = face_client.face.detect_with_stream(image2)

    if detected_faces2 != []:
        # List for the target face IDs (uuids)
        detected_faces_ids = []

        # Add the returned face's face ID
        source_image1_id = detected_faces1[0].face_id
        # Add the returned face's face ID
        source_image2_id = detected_faces2[0].face_id

        # Verification example for faces of the same person. The higher the confidence, the more identical the faces in the images are.
        # Since target faces are the same person, in this example, we can use the 1st ID in the detected_faces_ids list to compare.
        verify_result_same = face_client.face.verify_face_to_face(source_image1_id, source_image2_id)

        return verify_result_same.is_identical  
    else:
        return False      
#==============================================================
# Affiche plusieurs images dans une figure
features = ['Description', 'Tags', 'Adult', 'Objects', 'Faces']
def display_image2(urls, filenames):
    fig=plt.figure(figsize=(10,10))
    # blob_service_client = BlobServiceClient.from_connection_string(connect_str) 
    urls_ia=[]
    texts_ia=[]
    tags_ia=[]
    for i in range(len(urls)):
        title=''
        #a=fig.add_subplot(2,2,i+1)
        test=Image.open(BytesIO(requests.get(urls[i]).content))
        analysis = computervision_client.analyze_image(urls[i], visual_features=features)
        
        if (len(analysis.description.captions) == 0):
          title = 'No caption detected'
        else:
         for caption in analysis.description.captions:
            title = title + " '{}'\n(Confidence: {:.2f}%)".format(caption.text, caption.confidence * 100)
        # get objects
        if (len(analysis.objects) == 0):
            plt.xlabel('no objects detected')
        else:
        # Draw a rectangle around each object
         for object in analysis.objects:
            r = object.rectangle
            bounding_box = ((r.x, r.y), (r.x + r.w, r.y + r.h))
            draw = ImageDraw.Draw(test)
            draw.rectangle(bounding_box, outline='magenta', width=5)
            plt.annotate(object.object_property,(r.x, r.y), backgroundcolor='magenta')

        # Get faces
        if (len(analysis.faces) == 0):
         plt.ylabel('no faces detected')
        else: 
        # Draw a rectangle around each face
         for face in analysis.faces:
            r = face.face_rectangle
            bounding_box = ((r.left, r.top), (r.left + r.width, r.top + r.height))
            draw = ImageDraw.Draw(test)
            draw.rectangle(bounding_box, outline='lightgreen', width=5)
            annotation = 'Person aged approxilately {}'.format(face.age)
            plt.annotate(annotation,(r.left, r.top), backgroundcolor='lightgreen')
        plt.title(title)
        plt.imshow(test)
        plt.axis('off')

        # Enregistrement de l'image après analyse de l'ia en local et sur le blob
        ia_filename = "ia_" + filenames[i]
        plt.savefig("./static/uploads/" + ia_filename)
        url_ia = uploadImageOnBlob(ia_filename)
        urls_ia.append(url_ia)
        texts_ia.append(caption.text)

        # Récupération et génération d'une liste de tags
        tags_image_ia=[]
        for tag in analysis.description.tags:
            tags_image_ia.append(tag)
        tags_ia.append(tags_image_ia)
    
   
    for i in range(len(urls_ia)):
        # Enregistrement en bdd des infos des images
        addItems(urls_ia[i], texts_ia[i], tags_ia[i])

    return urls_ia, texts_ia
#==============================================================
def addItems(url, text, tags):
    # Add items to the container
    items_to_create = [createItems(url, text, tags)]
    print("- Item add : ", items_to_create)
    # <create_item>
    for family_item in items_to_create:
        container.create_item(body=family_item)
    # </create_item>
#==============================================================
# Méthode pour créer des items
def createItems(url, text, tags):
    item = {
        'id': str(uuid.uuid4()),
        'text': text,
        'url': url,
        'tags': tags
    }
    return item
#==============================================================
@app.route('/')
def home():
    return render_template("home.html")
#==============================================================
@app.route('/upload')
def uploadUrl():
    return render_template('upload.html')
#==============================================================
@app.route('/search')
def searchUrl():
    return render_template('search.html')
#==============================================================
@app.route('/list')
def listUrl():
    return render_template('list.html')
#==============================================================
@app.route('/', methods=['POST'])
def display_image():
    print("===================== START =====================")
    imgRecoFacial = request.files.getlist('imgRecoFacial')
    files = request.files.getlist('images')
    search = request.form.get('search')
    listImages = request.form.get('listImages')
    print("imgRecoFacial:", imgRecoFacial)
    print("files:", files)
    print("search:", search)
    print("affichage:", listImages)

    if imgRecoFacial != []:
        for imgIdentification in imgRecoFacial:
            if imgIdentification.filename != '':
                filenameReco = secure_filename(imgIdentification.filename)
                imgIdentification.save(os.path.join(app.config['UPLOAD_FOLDER'], filenameReco))
                valid = recoFacial(os.path.join(app.config['UPLOAD_FOLDER'], filenameReco))
                if valid:
                    return render_template("upload.html")
                else:
                    return render_template("home.html", noMatch=True)
            else:
                return render_template("home.html", noImage=True)

    if listImages != [] and listImages is not None:
        print("list")
        urls, descriptions = afficherList()
        return render_template("list.html", urls=urls, descriptions=descriptions)

    if search != [] and search is not None:
        print("search")
        urls, descriptions = searchInDb(search)
        return render_template("search.html", urls=urls, descriptions=descriptions)

    if files != []:
        print("files")
        filenames=[]
        urls=[]
        descriptions=[]
        for file in files:
            if file.filename != '':
                filename = secure_filename(file.filename)
                filenames.append(filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
                url = uploadImageOnBlob(filename)
                urls.append(url)

                # Get a description from the computer vision service
                description = computervision_client.describe_image(url)
                descriptions.append(description)

        urls, descriptions = display_image2(urls, filenames) 
        return render_template("upload.html", urls=urls, descriptions=descriptions)
#==============================================================
def searchInDb(search):
    # Query these items using the SQL query syntax. 
    # Specifying the partition key value in the query allows Cosmos DB to retrieve data only from the relevant partitions, which improves performance
    # <query_items>
    urls_ia=[]
    texts_ia=[]
    query = "SELECT * FROM c WHERE UPPER(c.text) LIKE UPPER('%" + search + "%')"

    items = list(container.query_items(
        query=query,
        enable_cross_partition_query=True
    ))

    request_charge = container.client_connection.last_response_headers['x-ms-request-charge']
    for item in items:
        urls_ia.append(item["url"])
        texts_ia.append(item["text"])
    # </query_items>
    return urls_ia, texts_ia
#==============================================================
def afficherList():
    # Query these items using the SQL query syntax. 
    # Specifying the partition key value in the query allows Cosmos DB to retrieve data only from the relevant partitions, which improves performance
    # <query_items>
    urls_ia=[]
    texts_ia=[]
    query = "SELECT * FROM c"

    items = list(container.query_items(
        query=query,
        enable_cross_partition_query=True
    ))

    request_charge = container.client_connection.last_response_headers['x-ms-request-charge']
    for item in items:
        urls_ia.append(item["url"])
        texts_ia.append(item["text"])
    # </query_items>
    return urls_ia, texts_ia
#==============================================================
def uploadImageOnBlob(filename):
    blob_client = blob_service_client.get_blob_client(container=AZURE_BLOB_NAME, blob=filename)
    
    with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), "rb") as data: 
        blob_client.upload_blob(data, overwrite=True)

    print("- Image : " + filename + " correctly upload on blob !")
    # Retourne l'url de l'image dans le blob
    return blob_client.url
#==============================================================

if __name__ == '__main__':
    app.run(debug = True)