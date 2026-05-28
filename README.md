# Detekcija prevara na kreditnim karticama pomoću dubokih neuronskih mreža

Ovaj projekat predstavlja implementaciju i komparativnu analizu dubokih neuronskih mreža (Feedforward MLP) za binarnu klasifikaciju transakcija na regularne i prevare. Kroz projekat je sproveden proces postepenog pronalaska optimalne arhitekture kroz tri različite eksperimentalne konfiguracije. Projekat je rađen u sklopu predmeta *Duboko učenje i neuronske mreže*.

## 1. Opis problema
Finansijske prevare sa kreditnim karticama nanose ogromne štete bankama i klijentima. Glavni izazov u rešavanju ovog problema pomoću veštačke inteligencije jeste **ekstremni debalans klasa** – prevare čine tek minimalni deo ukupnog broja transakcija (manje od 1%). Klasični algoritmi u ovim uslovima optimizuju ukupnu tačnost i potpuno ignorišu manjinsku klasu. Cilj ovog projekta je bio da se kroz eksperimente pronađe model koji maksimizuje detekciju prevara (Recall), uz održavanje operativne stabilnosti sistema kroz tehnike regularizacije.

## 2. Podaci
* **Izvor podataka:** Kaggle Credit Card Fraud Detection dataset.
* **Struktura:** Dataset sadrži ukupno 284,807 transakcija unutar dva dana. Sadrži 30 numeričkih atributa (V1-V28 dobijeni PCA transformacijom, `Time` i `Amount`) i ciljnu klasu `Class` (0 za regularne, 1 za prevare).
* **Analiza debalansa:** U datasetu ima svega 492 prevare (**0.172%**), dok su ostalo regularne transakcije.
* **Preprocesiranje:** Kolone `Amount` i `Time` su skalirane pomoću `StandardScaler`-a kako bi se sve karakteristike dovele na sličan opseg i ubrzala konvergencija. Podaci su podeljeni na trening (80%) i test set (20%) uz obavezno korišćenje **stratifikacije** (`stratify=y`) kako bi se u oba skupa očuvao identičan procenat prevara.

## 3. Arhitektura modela
Svi modeli u ovom projektu razvijeni su u **Keras/TensorFlow** okruženju kao sekvencijalne neuronske mreže sa topologijom levka (postepeno sažimanje informacija):
* **Ulazni sloj:** Prihvata 30 preprocesiranih karakteristika transakcije (`input_shape=(30,)`).
* **Skriveni sloj 1:** 32 neurona, `ReLU` nelinearna aktivaciona funkcija.
* **Skriveni sloj 2:** 16 neurona, `ReLU` nelinearna aktivaciona funkcija.
* **Izlazni sloj:** 1 neuron, `Sigmoid` aktivaciona funkcija (sabija izlaz u opseg [0, 1] što omogućava interpretaciju rezultata kao verovatnoće prevare).

## 4. Tok istraživanja i trening

U cilju pronalaska najboljeg modela, realizovane su tri različite konfiguracije:

### Model 1: Baseline model (Čista mreža)
Inicijalni model bez ikakvih modifikacija. Treniran kroz 5 epoha sa veličinom paketa (batch size) od 2048, koristeći `Adam` optimizator. Služi kao polazna osnova za dokazivanje problema debalansa.

### Model 2: Težinski model (Uvođenje Class Weights)
Zadržana je ista arhitektura, ali je u proces treninga uvedena penalizacija greške u realnom vremenu preko `compute_class_weight`. Klasa 1 (prevare) je dobila težinski koeficijent od **289.14**, čime je optimizator primoran da višestruko strože kažnjava model za svaku promašenu prevaru.

### Model 3: Finalni model (Težinski + Regularizacija)
Kako bi se ublažila nuspojava težinskog modela (preveliki broj lažnih uzbuna i osetljivost na šum), u finalnu arhitekturu su ubačene dve tehnike regularizacije:
* **Dropout sloj** sa stopom od 20% (0.2) između skrivenih slojeva (nasumično gašenje neurona tokom treninga sprečava ko-adaptaciju i teranje mreže da uči robusne šablone).
* **L2 regularizacija** uvedena direktno u Adam optimizator preko parametra `weight_decay=1e-4` za stabilizaciju težina.
Model je treniran kroz 10 epoha radi potpune konvergencije.

## 5. Analiza osetljivosti i hiperparametarska optimizacija

U ovom projektu, hiperparametarska optimizacija i analiza osetljivosti modela sprovedene su kroz kontrolisani eksperiment sa tri različite konfiguracije mreže. Cilj je bio utvrditi kako promena ključnih hiperparametara utiče na stabilnost učenja i krajnje metrike.

### Hiperparametarska optimizacija
Praćene su i optimizovane sledeće komponente:
1. **Strategija balansiranja klasa (`class_weight`):** Testiran je model bez balansiranja naspram modela sa otežanom funkcijom gubitka u odnosu **289.14 : 1** u korist klase prevara.
2. **Regularizacioni mehanizmi:** Upoređen je uticaj uvođenja **Dropout** slojeva (stopa 0.2) i L2 regularizacije (**Weight Decay** stope $1e-4$) unutar Adam optimizatora na suzbijanje informacionog šuma.
3. **Dužina treninga (Epohe):** Analiziran je napredak kroz promenu sa 5 epoha (brzi prototip) na 10 epoha (potpuna konvergencija).

## 6. Rezultati evaluacije
Nakon završnog testiranja na test setu koji sadrži 56,962 transakcije (od čega 98 stvarnih prevara), dobijeni su sledeći uporedni rezultati:

### Model 1 (Baseline)
* **Ukupna tačnost (Accuracy):** 1.00 (99.9%)
* **Recall (Odziv):** 0.71 (Model je promašio čak 26% stvarnih prevara)
* **Precision (Preciznost):** 0.82

### Model 2 (Težinski)
* **Ukupna tačnost (Accuracy):** 0.97
* **Recall (Odziv):** 0.92 (Ekstremni skok u detekciji)
* **Precision (Preciznost):** 0.06 (Model generiše veliki broj lažnih uzbuna)

### Model 3 (Finalni Regularizovani)
* **Ukupna tačnost (Accuracy):** 0.97
* **Recall (Odziv):** 0.94 (Optimalna osetljivost, ulovljeno 92 od 98 prevara)
* **Precision (Preciznost):** 0.06

**Matrica konfuzije za Finalni Model 3:**
* Stvarno regularnih, a model pogodio (True Negatives): **55,802**
* Regularnih, a model greškom proglasio prevarom (False Positives): **1,062**
* Prevara, a model ih promašio (False Negatives): **9**
* Prevara, a model ih uspešno ulovio (True Positives): **89**

## 7. Diskusija
Eksperimenti jasno oslikavaju fundamentalni kompromis (*Precision-Recall Trade-off*) u dubokom učenju. Uvođenjem težinskih koeficijenata u Modelu 2 i 3 svesno smo žrtvovali preciznost (pad na 7%) kako bismo maksimalno podigli odziv (skok na 94%). Sa poslovnog aspekta, trošak privremenog blokiranja kartice i slanja SMS verifikacije za 1,062 regularne transakcije je zanemarljiv u poređenju sa katastrofalnim finansijskim i pravnim gubicima koje bi banka pretrpela da je propustila 89 prevare koje je naš finalni model uspešno locirao.

## 8. Zaključak
Kroz proces inženjerskog prototipovanja uspešno je razvijena duboka neuronska mreža koja uspešno rešava problem detekcije anomalija u uslovima ekstremnog debalansa podataka. Kombinacijom `Class Weights` balansiranja i naprednih metoda regularizacije (`Dropout` i `Weight Decay`), Model 3 je ostvario vrhunski odziv od **94%** na neviđenim podacima, čime je cilj projekta u potpunosti ispunjen, a sistem dokazan kao stabilan i visoko osetljiv.

## Licenca
Ovaj projekat je licenciran pod MIT licencom - pogledajte [LICENSE](LICENSE) fajl za detalje.
