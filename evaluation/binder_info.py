BINDING_INFO = { # sequence, target_amino_acids, target_amino_indices, template, true_pKd, sequence source, binder source
    'EGFR' : (
        'LEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPS',
        ['S11', 'N12', 'K13', 'T15', 'Q16', 'L17', 'G18', 'S356', 'S440', 'G441'],
        [11, 12, 13, 15, 16, 17, 18, 356, 440, 441],
        'QVQLQQSGPGLVQPSQSLSITCTVSGFSLTNYGVHWVRQSPGKGLEWLGVIWSGGNTDYNTPFTSRLSISRDTSKSQVFFKMNSLQTDDTAIYYCARALTYYDYEFAYWGQGTLVTVSAGGGGSGGGGSGGGGSDILLTQSPVILSVSPGERVSFSCRASQSIGTNIHWYQQRTNGSPKLLIRYASESISGIPSRFSGSGSGTDFTLSINSVDPEDIADYYCQQNNNWPTTFGAGTKLELK',
        8.918238,
        'Adaptyv',
        'Cradle'
    ),
    'BBF-14' : (
        'MGTPLWALLGGPWRGTATYEDGTKVTLDYRYTRVSPDRLRADVTYTTPDGTTLEATVDLWKDANGVIRYHATYPDGTSADGTLTQLDADTLLATGTYDDGTKYTVTLTRVAPGSGWHHHHHH',
        [],
        [],
        'SPIQEEIQKKVRELLEKLIEYLEELKEKAKPPFKEKLEEVIEGLERLKEEVDKVQLNMNLIVFEGLEVDEEGRVWFIVKEMLHATTEEEALENMDKFLESWEKVFKELLEYHFEHNDTSPTFDFFLDFLWWQLYGEPMPKGSHHHHH',
        7.6798537,
        'EPFL',
        'EPFL (BindCraft)'
    ),
    'BHRF1': (
        'AYSTREILLALCIRDSRVHGNGTLHPVLELAARETPLRLSPEDTVVLRYHVLLEEIIERNSETFTETWNRFITHTEHVDLDFNSVFLEIFHRGDPSLGRALAWMAWCMHACRTLCCNQSTPYYVVDLSVRGMLEASEGLDGWIHQQGGWSTLIEDNIPG',
        ['F65', 'T74', 'E77', 'D82', 'S85', 'E89', 'R93', 'L98', 'G99', 'R100'],
        [65, 74, 77, 82, 85, 89, 93, 98, 99, 100],
        'MPSAFQIGLALVAAALDRALPEPYRGLALAIAAELSGLPEEELRRLVEAAEKAASADLPFEQQVGLALARIAAAVAGVGLARRAPSLPPEELLAAIREAIEEGGRIAAKALTRSGALEPVLAELP',
        8.070581,
        '',
        'AlphaProteo'
    ),
    'Cas9': (
        'MDKKYSIGLDIGTNSVGWAVITDEYKVPSKKFKVLGNTDRHSIKKNLIGALLFDSGETAEATRLKRTARRRYTRRKNRICYLQEIFSNEMAKVDDSFFHRLEESFLVEEDKKHERHPIFGNIVDEVAYHEKYPTIYHLRKKLVDSTDKADLRLIYLALAHMIKFRGHFLIEGDLNPDNSDVDKLFIQLVQTYNQLFEENPINASGVDAKAILSARLSKSRRLENLIAQLPGEKKNGLFGNLIALSLGLTPNFKSNFDLAEDAKLQLSKDTYDDDLDNLLAQIGDQYADLFLAAKNLSDAILLSDILRVNTEITKAPLSASMIKRYDEHHQDLTLLKALVRQQLPEKYKEIFFDQSKNGYAGYIDGGASQEEFYKFIKPILEKMDGTEELLVKLNREDLLRKQRTFDNGSIPHQIHLGELHAILRRQEDFYPFLKDNREKIEKILTFRIPYYVGPLARGNSRFAWMTRKSEETITPWNFEEVVDKGASAQSFIERMTNFDKNLPNEKVLPKHSLLYEYFTVYNELTKVKYVTEGMRKPAFLSGEQKKAIVDLLFKTNRKVTVKQLKEDYFKKIECFDSVEISGVEDRFNASLGTYHDLLKIIKDKDFLDNEENEDILEDIVLTLTLFEDREMIEERLKTYAHLFDDKVMKQLKRRRYTGWGRLSRKLINGIRDKQSGKTILDFLKSDGFANRNFMQLIHDDSLTFKEDIQKAQVSGQGDSLHEHIANLAGSPAIKKGILQTVKVVDELVKVMGRHKPENIVIEMARENQTTQKGQKNSRERMKRIEEGIKELGSQILKEHPVENTQLQNEKLYLYYLQNGRDMYVDQELDINRLSDYDVDHIVPQSFLKDDSIDNKVLTRSDKNRGKSDNVPSEEVVKKMKNYWRQLLNAKLITQRKFDNLTKAERGGLSELDKAGFIKRQLVETRQITKHVAQILDSRMNTKYDENDKLIREVKVITLKSKLVSDFRKDFQFYKVREINNYHHAHDAYLNAVVGTALIKKYPKLESEFVYGDYKVYDVRKMIAKSEQEIGKATAKYFFYSNIMNFFKTEITLANGEIRKRPLIETNGETGEIVWDKGRDFATVRKVLSMPQVNIVKKTEVQTGGFSKESILPKRNSDKLIARKKDWDPKKYGGFDSPTVAYSVLVVAKVEKGKSKKLKSVKELLGITIMERSSFEKNPIDFLEAKGYKEVKKDLIIKLPKYSLFELENGRKRMLASAGELQKGNELALPSKYVNFLYLASHYEKLKGSPEDNEQKQLFVEQHKHYLDEIIEQISEFSKRVILADANLDKVLSAYNKHRDKPIREQAENIIHLFTLTNLGAPAAFKYFDTTIDRKRYTSTKEVLDATLIHQSITGLYETRIDLSQLGGD',
        ['T360'],
        [360],
        'SEEEKKEILYFIMEKLFDLDFNFKWPRNSPEEYTKAIEEFKAEVAKIVLETKEKFPEISPEELVELLEEAVYRVHRITHHWASYYVAREVIYELKKLKEKGWKAIEEYTESIISKV',
        6.422508,
        'UniProt',
        'EPFL (BindCraft)'
    ),
    'IL-7Ralpha': (
        'ESGYAQNGDLEDAELDDYSFSCYSQLEVNGSQHSLTCAFEDPDVNTTNLEFEICGALVEVKCLNFRKLQEIYFIETKKFLLIGKSNICVKVGEKSLTCKKIDLTTIVKPEAPFDLSVVYREGANDFVVTFNTSHLQKKYVKVLMHDVAYRQEKDENKWTHVNLSSTKLTLLQRKLQPAAMYEIKVRSIPDHYFKGFWSEWSPSYYFRTPEINNSSGEMD',
        ['V58', 'L80', 'Y139'],
        [58, 80, 139],
        'MTKVEEAKELVDKIMEAAKAKDLEKVNKLRTEFFELVNSLSLEEAEEVRKYADKKGEEWYKEQL',
        10.086186,
        '',
        'AlphaProteo'
    ),  
    'MBP': (
        'KIEEGKLVIWINGDKGYNGLAEVGKKFEKDTGIKVTVEHPDKLEEKFPQVAATGDGPDIIFWAHDRFGGYAQSGLLAEITPDKAFQDKLYPFTWDAVRYNGKLIAYPIAVEALSLIYNKDLLPNPPKTWEEIPALDKELKAKGKSALMFNLQEPYFTWPLIAADGGYAFKYENGKYDIKDVGVDNAGAKAGLTFLVDLIKNKHMNADTDYSIAEAAFNKGETAMTINGPWAWSNIDTSKVNYGVTVLPTFKGQPSKPFVGVLSAGINAASPNKELAKEFLENYLLTDEGLEAVNKDKPLGAVALKSYEEELAKDPRIAATMENAQKGEIMPNIPQMSAFWYAVRTAVINAASGRQTVDEALKDAQTRITK',
        ['Y90', 'P91', 'F92', 'Y171', 'Y176', 'P315', 'M321', 'I329'],
        [90, 91, 92, 171, 176, 315, 321, 329],
        'SQVQLVESGGGSVQAGGSLRLSCVASGDIKYISYLGWFRQAPGKEREGVAALYTSTGRTYYADSVKGRFTVSLDNAKNTVYLQMNSLKPEDTALYYCAAAEWGSQSPLTQWFYRYWGQGTQVTVSA',
        7.617983,
        '',
        'Zimmermann et al. (2018)'
    ),
    'PD-L1': (
        'FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYNKINQRILVVDPVTSEHELTCQAEGYPKAEVIWTSSDHQVLSGKTTTTNSKREEKLFNVTSTLRINTTTNEIFYCTFRRLDPEENHTAELVIPELPLAHPPNERT',
        ['I54', 'Y56', 'E58', 'N63', 'Q66','V76','R113','M115', 'S117', 'A121', 'Y123'],
        [54, 56, 58, 63, 66, 76, 113, 115, 117, 121, 123],
        'SAEEKILANLEAMKAKALAAKTEEEKLFYAKALLAVAISYAIRGDYELARRAAELAVEVIKSLSKEEQKKVMDFLINIIKNITDPEDREKAIELAIAIAERLDEEVREEALKKIEELKKE',
        10.375718,
        'UniProt',
        'AlphaProteo'
    ),
    'USP15': (
        'MAEGGAADLDTQRSDIATLLKTSLRKGDTWYLVDSRWFKQWKKYVGFDSWDKYQMGDQNVYPGPIDNSGLLKDGDAQSLKEHLIDELDYILLPTEGWNKLVSWYTLMEGQEPIARKVVEQGMFVKHCKVEVYLTELKLCENGNMNNVVTRRFSKADTIDTIEKEIRKIFSIPDEKETRLWNKYMSNTFEPLNKPDSTIQDAGLYQGQVLVIEQKNEDGTWPRGPSTPKSPGASNFSTLPKISPSSLSNNYNNMNNRNVKNSNYCLPSYTAYKNYDYSEPGRNNEQPGLCGLSNLGNTCFMNSAIQCLSNTPPLTEYFLNDKYQEELNFDNPLGMRGEIAKSYAELIKQMWSGKFSYVTPRAFKTQVGRFAPQFSGYQQQDCQELLAFLLDGLHEDLNRIRKKPYIQLKDADGRPDKVVAEEAWENHLKRNDSIIVDIFHGLFKSTLVCPECAKISVTFDPFCYLTLPLPMKKERTLEVYLVRMDPLTKPMQYKVVVPKIGNILDLCTALSALSGIPADKMIVTDIYNHRFHRIFAMDENLSSIMERDDIYVFEININRTEDTEHVIIPVCLREKFRHSSYTHHTGSSLFGQPFLMAVPRNNTEDKLYNLLLLRMCRYVKISTETEETEGSLHCCKDQNINGNGPNGIHEEGSPSEMETDEPDDESSQDQELPSENENSQSEDSVGGDNDSENGLCTEDTCKGQLTGHKKRLFTFQFNNLGNTDINYIKDDTRHIRFDDRQLRLDERSFLALDWDPDLKKRYFDENAAEDFEKHESVEYKPPKKPFVKLKDCIELFTTKEKLGAEDPWYCPNCKEHQQATKKLDLWSLPPVLVVHLKRFSYSRYMRDKLDTLVDFPINDLDMSEFLINPNAGPCRYNLIAVSNHYGGMGGGHYTAFAKNKDDGKWYYFDDSSVSTASEDQIVSKAAYVLFYQRQDTFSGTGFFPLDRETKGASAATGIPLESDEDSNDNDNDIENENCMHTN',
        [],
        [],
        'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG',
        8.0,
        '',
        ''
    )

}
