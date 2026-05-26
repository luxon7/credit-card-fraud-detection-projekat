# Detekcija prevara na kreditnim karticama pomoću dubokih neuronskih mreža

Ovaj projekat predstavlja implementaciju duboke neuronske mreže (Feedforward MLP) za binarnu klasifikaciju transakcija na regularne i prevare. Projekat je rađen u sklopu predmeta *Duboko učenje i neuronske mreže*.

## 1. Opis problema
Finansijske prevare sa kreditnim karticama nanose ogromne štete bankama i klijentima. Glavni izazov u rešavanju ovog problema pomoću veštačke inteligencije jeste **ekstremni debalans klasa** – prevare čine tek minimalni deo ukupnog broja transakcija, što klasične algoritme vodi ka pogrešnom zaključivanju. Cilj ovog modela je da maksimizuje detekciju prevara (Recall) uz održavanje stabilnosti sistema.

## 2. Podaci
* **Izvor podataka:** Kaggle Credit Card Fraud Detection dataset.
* **Struktura:** Dataset sadrži ukupno 284,807 transakcija unutar dva dana. Sadrži 30 numeričkih atributa (V1-V28 dobijeni PCA transformacijom, `Time` i `Amount`) i ciljnu klasu `Class` (0 za regularne, 1 za prevare).
* **Analiza debalansa:** U datasetu ima svega 492 prevare (**0.172%**), dok su ostalo regularne transakcije.
* **Preprocesiranje:** Kolone `Amount` i `Time` su skalirane pomoću `StandardScaler`-a kako bi se sve karakteristike dovele na sličan opseg. Podaci su podeljeni na trening (80%) i test set (20%) uz korišćenje **stratifikacije** (`stratify=y`) kako bi se očuvao procenat prevara u oba seta.

## 3. Arhitektura modela
Model je izgrađen pomoću Keras/TensorFlow platforme kao sekvencijalna neuronska mreža:
* **Ulazni sloj:** Prihvata 30 karakteristika transakcije.
* **Skriveni sloj 1:** 32 neurona, `ReLU` aktivaciona funkcija.
* **Regularizacija:** `Dropout` sloj sa stopom od 20% (0.2) radi sprečavanja overfitting-a.
* **Skriveni sloj 2:** 16 neurona, `ReLU` aktivaciona funkcija.
* **Izlazni sloj:** 1 neuron, `Sigmoid` aktivaciona funkcija (daje verovatnoću prevare u opsegu od 0 do 1).

## 4. Trening
* **Optimizator:** `Adam` (learning rate = 0.001)
* **Funkcija gubitka (Loss):** `BinaryCrossentropy`
* **Veličina paketa (Batch size):** 2048
* **Broj epoha:** 10
* **Strategija balansiranja:** Izračunate su težine klasa (`Class Weights`) gde je klasa 1 dobila penalizaciju od **289.14**, što je primoralo mrežu da obrati posebnu pažnju na retke uzorke prevara.

## 5. Analiza osetljivosti i hiperparametarska optimizacija
Tokom razvoja testirane su različite konfiguracije modela:
1. Bez težina klasa: Model je postizao ukupnu tačnost (Accuracy) od 99.9%, ali je Recall za prevare bio blizu 0% (model je ignorisao prevare).
2. Sa uvođenjem `Class Weights`: Izuzetno povećanje Recall-a na preko 90%, uz očekivani pad Precision-a.
3. Dodavanje `Dropout(0.2)` sloja: Stabilizovan gubitak (Loss) na validacionom setu i sprečeno preprilagođavanje specifičnim šablonima iz trening seta.

## 6. Rezultati evaluacije
Nakon završnog testiranja na test setu (56,962 transakcije), model je ostvario sledeće rezultate:

* **Ukupna tačnost (Accuracy):** 98%
* **Recall (Klasa 1 - Prevare):** 0.91 (Model uspešno detektuje 91% svih prevara)
* **Precision (Klasa 1 - Prevare):** 0.07

**Matrica konfuzije:**
* Stvarno regularnih, a model pogodio (True Negatives): **55,682**
* Regularnih, a model greškom proglasio prevarom (False Positives): **1,182**
* Prevara, a model ih promašio (False Negatives): **9**
* Prevara, a model ih uspešno ulovio (True Positives): **89**

## 7. Diskusija
Iako model generiše određen broj lažnih uzbuna (1,182 transakcije gde je korisnik zapravo regularno plaćao), u realnom poslovnom sistemu banke ovo je prihvatljiv rizik. Trošak blokiranja kartice i slanja verifikacionog SMS-a je zanemarljiv u odnosu na finansijsku i pravnu štetu koju bi banka pretrpela ako bi propustila 89 prevara koje je naš model uspešno detektovao.

## 8. Zaključak
Implementirana duboka neuronska mreža uspešno rešava problem detekcije prevara na ekstremno neuravnoteženim podacima. Korišćenjem težinskih koeficijenata klasa i slojeva za regularizaciju postignut je robustan model visoke osetljivosti (Recall = 91%), čime je cilj projekta u potpunosti ispunjen.

## Licenca
Ovaj projekat je licenciran pod MIT licencom - pogledajte [LICENSE](LICENSE) fajl za detalje.
