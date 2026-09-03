# Utvärdering: naivt mot agentiskt läge

30 frågor, fyra kategorier. Samma systemprompt i båda lägena – 
skillnaden ligger enbart i hur underlaget hämtas.

| Kategori | Läge | n | Citatprecision | Groundedness | Partier | Utdrag | Avstod | Sek | Anrop |
|---|---|---|---|---|---|---|---|---|---|
| Smal faktafråga | naivt | 7 | 98 % | 100 % | 3.7 | 8.0 | 0 % | 28 | 1.0 |
| Smal faktafråga | agentiskt | 7 | 99 % | 99 % | 6.4 | 21.6 | 0 % | 55 | 3.0 |
| Bred partijämförelse | naivt | 8 | 98 % | 99 % | 4.8 | 8.0 | 0 % | 29 | 1.0 |
| Bred partijämförelse | agentiskt | 8 | 94 % | 100 % | 8.0 | 24.0 | 0 % | 62 | 3.5 |
| Personfråga | naivt | 8 | 100 % | 100 % | 3.9 | 8.0 | 75 % | 16 | 1.0 |
| Personfråga | agentiskt | 8 | 100 % | 99 % | 1.0 | 16.5 | 0 % | 46 | 3.0 |
| Obesvarbar | naivt | 7 | 83 % | 100 % | 4.3 | 8.0 | 43 % | 8 | 1.0 |
| Obesvarbar | agentiskt | 7 | 94 % | 99 % | 5.0 | 7.6 | 57 % | 24 | 3.0 |

## Totalt

| Läge | Citatprecision | Groundedness | Sek/fråga | Anrop/fråga |
|---|---|---|---|---|
| naivt | 96 % | 100 % | 20 | 1.0 |
| agentiskt | 97 % | 99 % | 47 | 3.1 |

## Så mäts det

- **Citatprecision** mäts mekaniskt: citaten `[Namn, datum]` plockas ur svaret med
  reguljärt uttryck och jämförs mot metadatan i de utdrag som faktiskt skickades.
  Ingen modell inblandad, så måttet kan inte drabbas av samma fel det ska upptäcka.
- **Groundedness** bedöms av Claude, som får frågan, svaret och källorna och prövar
  varje påstående för sig. Att korrekt återge vad någon *sade* räknas som stött -
  vi mäter trohet mot källan, inte sanningshalt.
- **Partier** och **Utdrag** är genomsnitt över frågorna i kategorin.
- **Avstod** = andel svar som i huvudsak säger att underlaget inte räcker.

## Förbehåll

- **Kategorin "obesvarbar" mäter inte vad den var tänkt att mäta.** Bara 3 av 7
  frågor saknade verkligen underlag (Melodifestivalen, Allsvenskan, Zlatan). De
  övriga - mjölkpris, avståndet Stockholm-Göteborg, Sveriges högsta byggnad,
  månadskort - *förekommer* i anförandena, eftersom ledamöter använder vardagsfakta
  retoriskt. Systemet svarade korrekt genom att tillskriva uppgiften den som sade
  den. Avstod-siffran i den raden är därför utspädd, inte ett tecken på gissningar.
- **Naivt läge har inget talarfilter.** Det är avsiktligt: filtret är ett verktyg
  agenten kan välja att använda. Det förklarar hela skillnaden i personraden.
- **Kolumnen Partier är bara meningsfull för breda frågor.** För personfrågor är
  1.0 det korrekta värdet, inte ett dåligt.
- **Bedömaren är samma modellfamilj som den som svarar.** Groundedness-siffran bör
  läsas som en indikation, inte ett oberoende facit.
