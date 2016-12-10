library(readr)
library(FactoMineR)
eco <- read_csv("C:/Users/jerem/Google Drive/Education/McGill_/COMP551/ProjectR/ecology.csv", 
                col_types = cols(timestamp = col_datetime(format = "%Y-%m-%d %H:%M:%S")))


ecology <- eco[,c(3,4,5,13,16,17,18,22,23)]

ecology['gender'] = NA
ecology[ecology$'tag-local-identifier' %in% c(765,767,770,774,1021,1190,1273),]['gender'] = 'Female'
ecology[ecology$'tag-local-identifier' %in% c(766,769,771,775,1022,1191,1274,1396,1397,1403),]['gender'] = 'Male'

alleco <- ecology
ecology = na.omit(ecology)

ecology['shell'] = NA
ecology['altirange'] = NA
ecology[ecology$'tag-local-identifier' == 765,]['shell'] = 90.3
ecology[ecology$'tag-local-identifier' == 765,]['altirange'] = 36
ecology[ecology$'tag-local-identifier' == 766,]['shell'] = 118.5
ecology[ecology$'tag-local-identifier' == 766,]['altirange'] = 279
ecology[ecology$'tag-local-identifier' == 767,]['shell'] = 89.5
ecology[ecology$'tag-local-identifier' == 767,]['altirange'] = 144
ecology[ecology$'tag-local-identifier' == 769,]['shell'] = 109.5
ecology[ecology$'tag-local-identifier' == 769,]['altirange'] = 66
ecology[ecology$'tag-local-identifier' == 770,]['shell'] = 100
ecology[ecology$'tag-local-identifier' == 770,]['altirange'] = 196
ecology[ecology$'tag-local-identifier' == 771,]['shell'] = 151
ecology[ecology$'tag-local-identifier' == 771,]['altirange'] = 197
ecology[ecology$'tag-local-identifier' == 774,]['shell'] = 105
ecology[ecology$'tag-local-identifier' == 774,]['altirange'] = 315
ecology[ecology$'tag-local-identifier' == 775,]['shell'] = 142
ecology[ecology$'tag-local-identifier' == 775,]['altirange'] = 267
ecology[ecology$'tag-local-identifier' == 1021,]['shell'] = 93
ecology[ecology$'tag-local-identifier' == 1021,]['altirange'] = 272
ecology[ecology$'tag-local-identifier' == 1022,]['shell'] = 150.2
ecology[ecology$'tag-local-identifier' == 1022,]['altirange'] = 279
ecology[ecology$'tag-local-identifier' == 1190,]['shell'] = 96.1
ecology[ecology$'tag-local-identifier' == 1190,]['altirange'] = 22
ecology[ecology$'tag-local-identifier' == 1191,]['shell'] = 129.6
ecology[ecology$'tag-local-identifier' == 1191,]['altirange'] = 265
ecology[ecology$'tag-local-identifier' == 1273,]['shell'] = 113.8
ecology[ecology$'tag-local-identifier' == 1273,]['altirange'] = 194
ecology[ecology$'tag-local-identifier' == 1274,]['shell'] = 126.8
ecology[ecology$'tag-local-identifier' == 1274,]['altirange'] = 171
ecology[ecology$'tag-local-identifier' == 1396,]['shell'] = 139.2
ecology[ecology$'tag-local-identifier' == 1396,]['altirange'] = 303
ecology[ecology$'tag-local-identifier' == 1397,]['shell'] = 149.4
ecology[ecology$'tag-local-identifier' == 1397,]['altirange'] = 305
ecology[ecology$'tag-local-identifier' == 1403,]['shell'] = 145
ecology[ecology$'tag-local-identifier' == 1403,]['altirange'] = 277

reduced <- ecology[,c(9,10,11,12)]

unireduced <- reduced[!duplicated(reduced), ]
unireduced$gender <- as.factor(unireduced$gender)

meansMonth <- eco
meansMonth$month <-  format(eco$timestamp, "%b")

meansMonth <- as.data.frame(meansMonth)
names(meansMonth)[names(meansMonth)=="height-above-ellipsoid"] <- "height"
names(meansMonth)[names(meansMonth)=="location-long"] <- "long"
names(meansMonth)[names(meansMonth)=="location-lat"] <- "lati"
names(meansMonth)[names(meansMonth)=="individual-local-identifier"] <- "name"

altim <- aggregate( height ~ month+name, meansMonth, mean )
longi <- aggregate( long ~ name, meansMonth, mean )
lati <- aggregate( lati ~ name, meansMonth, mean )


for (m in unique(format(eco$timestamp, "%b"))) {
  print(m)
  for(i in 1:nrow(unireduced)) {
    name <- unireduced[[i,1]]
    height <- altim[altim$month == m & altim$name ==name,]$height
    lo <- longi[longi$name == name,]$long
    if(lo < -91){
      unireduced[unireduced$`individual-local-identifier` == name,'island'] = 1
    }else if(lo < -90){
      unireduced[unireduced$`individual-local-identifier` == name,'island'] = 2
    }else{
      unireduced[unireduced$`individual-local-identifier` == name,'island'] = 3
    }
    
    if(length(height)>0){
      unireduced[unireduced$`individual-local-identifier` == name,m] = height
    }
    
  }
  unireduced[m] <- as.numeric(unlist(unireduced[m]))
}

unireduced = na.omit(unireduced)
scaled <- unireduced

scaled[,-c(1,2,3,4,5)] <- t(scale(t(scaled[,-c(1,2,3,4,5)])))
scaled <- as.data.frame((scaled))

shell <- aggregate( shell ~ gender, unireduced, mean )
var <- aggregate( shell ~ gender, unireduced, sd )


res = MFA(unireduced[,c(3,4,6,7,8,9,10,11,12,13,14,15,16,17)], 
          group = c(1,1,12), type = c('c','c','c'), 
          name.group = c('Shell.size', 'Alt.range','Altitude'))

plot(res, habillage='ind', col.hab=scaled[,c(2)], title='')
plot(res, choix='var', title='')
res.hcpc = HCPC(res)
plot(res.hcpc, choice='map', title='', draw.tree=F, centers.plot=F)


res = MFA(unireduced[,c(2,3,4,6,7,8,9,10,11,12,13,14,15,16,17)], 
          group = c(1,1,1,12), type = c('n','c','c','c'), 
          name.group = c('Gender','Shell.size', 'Alt.range','Altitude'))

plot(res, habillage='ind', col.hab=scaled[,c(2)], title='')
plot(res, choix='var', title='')

res = PCA(scaled[,c(3,4,6,7,8,9,10,11,12,13,14,15,16,17)])
res.hcpc = HCPC(res)
plot(res.hcpc)
plot(res, habillage='ind', col.hab=scaled[,c(2)], title='')

