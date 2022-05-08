Ce projet a été réalisé par Alae Kichi et Jinan Loubani. La commande permettant de lancer correctement le script ```main.py``` est: python main.py --weights='weights.pth' --test_data='test_script.csv'

# Projet AI Frameworks

Dans ce projet, vous allez travailler sur des données<sup>[1](#myfootnote1)</sup>issues du site [Food.com](https://www.food.com/), un célèbre site de recettes de cuisine.   
![](img/food.png)
Les données, disponibles [ici](https://drive.google.com/drive/folders/18JyoxTIrIH2s2wG6HtxGiKsdFtGSfUWm?usp=sharing), contiennent des informations sur des recettes de cuisines ainsi que des interactions de plusieurs utilisateurs avec les recettes.   

## Consignes:
Les parties 1, 2 et 3 sont à réaliser dans un même notebook.
### Partie 1: Recommandations simples:
Dans __un notebook__
*   Présentez plusieurs stratégies de recommandation  de recettes:
    *   Par popularité
    *   Selon les étapes de la recette (colonne steps)
    *   Selon la description de la recette 

    Pour chacune de ces méthodes montrez quelques exemples des recommandation  obtenues.

### Partie 2: Analyse de sentiments:
Dans __un notebook__
*   À partir de la note donnée par les utilisateurs definissez une nouvelle variable sur le sentiment positif ou négatif d'un utilisateur vis-à-vis d'une recette.
Par exemple, toutes les notes inférieures à 3 sont négatives et celles supérieurs sont positives.
* En vous inspirant du TP [_Text Cleaning and Vectorization_](https://colab.research.google.com/github/wikistat/AI-Frameworks/blob/master/Text/1_cleaning_vectorization.ipynb) et du TP [_Interpretability in Machine Learning_](https://colab.research.google.com/github/wikistat/AI-Frameworks/blob/website/code/interpretability/TP_interpretability_solution.ipynb), entrainez un modèle à prédire si un utilisateur a aimé ou non une recette à partir de son commentaire et utilisez la méthode LIME pour visualiser les mots permettant de justifier la décision de votre modèle.  
Montrez quelques exemples de prédiction et de visualisations des mots importants.

### Partie 3: Neural Collaborative Filtering:
* Reprenez la classe ```NCF``` (Neural Collaborative Filtering )présente dans le [TP sur les systèmes de recommendations](https://colab.research.google.com/github/wikistat/AI-Frameworks/blob/website/code/recommender_systems/INSA_Reco_solution.ipynb) pour entrainer un modèle de Neural Collaborative Filtering à prédire les notes d'un utilisateur.
* Entrainez votre réseau sur les données train et testez le sur les données de test (calculez la Mean Absolute Error sur les données de test).
* Enregistrez les poids de votre réseau dans un fichier ```weight.pth```


### Partie 4 Scripts et Github:
*   Dans fichier ```model.py```, redefinissez la classe ```NCF``` (Neural Collaborative Filtering )présente dans le [TP sur les systèmes de recommandations](https://colab.research.google.com/github/wikistat/AI-Frameworks/blob/website/code/recommender_systems/INSA_Reco_solution.ipynb).
* Dans un fichier ```main.py```, implémentez un code permettant de prédire les notes d'un utilisateur.
Ce programme ne fera pas d'entrainement mais récupérera les poids su réseau que vous aurez entraîné dans votre notebook.
Ce fichier sera exécuté comme un script et devra:
    * Récupérer les poids du réseau à partir d'un chemin donné en argument du job.
    * Récupérer le chemin d'un fichier de test contenant 10 interactions (```test_script.csv```) en arguments de la commande qui exécutera le script 
    *   Afficher les prédictions pour les 10 interactions du dataset de test.  

Pour le livrable:

*   Faites un fork de ce repo Git.
*   Modifiez le pour qu'il contienne le notebook des parties 1,2 et 3.
*   Rajoutez les fichiers ```model.py``` et ```main.py```
*   Modifiez le readme pour qu'il affiche votre nom et la commande permettant de lancer correctement le script ```main.py``` (ne mettez pas les fichiers de données dans le repo github!)

<a name="myfootnote1">1</a>: Les données ont été récoltées  pour l'article suivant:  
 [Generating Personalized Recipes from Historical User Preferences
Bodhisattwa Prasad Majumder*, Shuyang Li*, Jianmo Ni, Julian McAuley
EMNLP, 2019](https://www.aclweb.org/anthology/D19-1613/)