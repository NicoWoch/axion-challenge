## Notatka z zagadnienia

Po wykładzie zobaczyłem film od freeCodeCamp.org wysłany na kanale `baza-wiedzy`. Po nim zrozumiałem więcej co się działo na wykładzie i dodatkowo dowiedziałem się o zjawisku zwanym **Phase Kickback**.

Nie spodziewałem się tego że bramka kontrolowana (np. CNOT) potrafi nie zmienić stanu qubita target, a zmienić stan control.

CNOT (z bitem kontrolnym pierwszym) zamienia stan $\ket{+-}$ w stan $\ket{--}$. Czyli zmienia się faza bitu kontrolnego, a bit target pozostaje bez zmian.

Jest to nieintuicyjne z punktu widzenia komputera klasycznego, ale matma się zgadza.

Teraz już wiem że po zastosowaniu bramki kontrolowanej na eigenvector tej bramki może dojść do zmiany fazy controla.
