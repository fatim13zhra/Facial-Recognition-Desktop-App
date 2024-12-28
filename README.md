# Application de Reconnaissance Faciale - Lisez-moi (README)

Cette application utilise la bibliothèque OpenCV, tkinter et face_recognition pour la reconnaissance faciale en utilisant différents modèles. Elle permet de capturer des images en direct depuis la caméra, de détecter les visages et de les reconnaître en les comparant à des visages enregistrés.

## Fonctionnalités
- Capture d'images en direct depuis la caméra.
- Détection de visages en utilisant différents modèles de détection :
  - Haar Cascade
  - HOG (Histogram of Oriented Gradients)
  - HOG sécurisé
- Reconnaissance faciale en comparant les visages détectés avec des visages enregistrés.
- Affichage des résultats de reconnaissance avec le nom de la personne détectée et le pourcentage de précision.
- Marquage de la présence en enregistrant les données dans un fichier CSV.
- Interface utilisateur personnalisable avec différents modes d'apparence : Clair, Foncé et Système.
- Option de mise à l'échelle de l'interface utilisateur pour ajuster la taille des éléments.

## Prérequis
- Python 3.x
- Les bibliothèques qui doivent être installées sont dns le fichier requirements.

## Comment exécuter l'application
1. Vérifiez que vous avez installé toutes les bibliothèques nécessaires mentionnées précédemment. 
2. Lancez le fichier Python GUI. 
3. Saisissez votre nom dans le champ "Nouvel utilisateur" et appuyez sur "Entrée". 
4. Ensuite, cliquez sur "Ouvrir la caméra" et sélectionnez le modèle de votre choix.

## Utilisation de l'application
- Sélectionnez le modèle de détection souhaité en cliquant sur les boutons correspondants dans la barre latérale.
- Pour capturer des images et générer des ensembles de données pour la reconnaissance faciale, saisissez le nom de la personne dans la zone de saisie et cliquez sur le bouton "Générer l'ensemble de données".
- Lorsque vous exécutez l'application, la caméra s'ouvrira et commencera à détecter les visages en temps réel.
- Les visages détectés seront comparés aux visages enregistrés et le nom de la personne et la précision de la reconnaissance seront affichés.
- Si la personne détectée n'est pas dans la liste des visages enregistrés, elle sera enregistrée dans un fichier CSV avec la date et l'heure de la détection.
- Vous pouvez modifier le mode d'apparence de l'interface en sélectionnant l'option correspondante dans la barre latérale.
- Vous pouvez également ajuster la mise à l'échelle de l'interface en sélectionnant une valeur dans l'option correspondante dans la barre latérale.

C'est tout ! Vous pouvez maintenant utiliser l'application de reconnaissance faciale pour détecter et reconnaître des visages en direct depuis votre caméra.

**Note :** Assurez-vous d'avoir les autorisations appropriées pour accéder à la caméra de votre appareil.

Veuillez noter que cette application peut nécessiter des modifications supplémentaires en fonction de votre cas d'utilisation et de vos besoins spécifiques.