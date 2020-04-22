import os
from os.path import join
import sys
print(sys.version)
import numpy as np
import flask

from tqdm import tqdm
import dash
print(dash.__file__)
import dash_core_components as dcc
import pandas as pd
import dash_html_components as html
from dash.dependencies import Input, Output
from base64 import b64encode
from rdkit.Chem import Draw
from rdkit import Chem

git_icon = 'data:image/webp;base64,UklGRuQaAABXRUJQVlA4TNgaAAAv/8F/EP8HJIT/59WICUjcwx8AQKqdZv839+YmNw7B7S1evG5QihR3d5fiUiTFHty1uD/AU+qGVtBCDYpDcIcEQkhCBIjd3Dt/JCT3nJ3ZM3nl84no/wSAdXaUeL1xl8ET5q/7cs/Bv89di4xNSH6e5vJ4XGnPkxNiI6+d+/vgni/XzZ8wuEvj10s4IO+40Outhy/44rdLcR40tSfu0uEvFgxv/VrBvBt7mUZDluy8+ByVfx6xY8mQhmXseSn2im0nf3E2FYlNOfv55DYV7XkezrcGrTuegoQ/P7b2o7eceRT26gM2nXchi67zmwZUs+cthDSbcygJmU06OKdpSN5AgTZLT2Ui05knl7YOkz3/hgvPeJB5z5kFDZxCVz18fwpqYsq+8dWkzdl09V3UzDurmjjFrEDfHc9QS5/u6BMmYEUG73ehxmbsG1REtAoPO+pG7c08MrSQUAX3+sWFmuz6uWewOPm0+CYFtfr51818JKnSgoeo4Q/mVxSi4IF/o7b/OSBIfqqvSUatT1pdTXT8uv+BFvD3bn5SU3z2Y7SIMbOKScyb29PRQqZ/9oaw2Nv/iZbzj3Y2OfEbeA0t6ZX+fjISEv4ALWvUuBD5CJuVgJb2ycz8slFgThJa3sRZYXJRcG4yWuKk2QVkImR6MlrmpKnB8uAcE4uWOma0UxZ8BtxHy32vn10Qml1ES36hsRS8sh8t+y/VJKDYZjda+MyNRa2eb3gyWvyksb6WrslVFMDLDa1b2Z0ohD+Utma+U1JQDFMmOixY7UsoihdqWq2wTR4URs/6/JaqSwwKZHQH61T0RxTK74pYpJ7xKJax3a1Q8T0omruKWZ7O8SiccR2tTf4vUEC357MwDSJRRO/Vtyp+Sz0opJ5Fvpak4mkU1BPlLUjvpyiqyd2tRvB2FNdtQZai+lUU2MtVLETP5yiyz7pZBecGFNu1fpag9CkU3BMvWYD6sSi6j+tq32gXCq9rhN45t6IA/9upccX/QRE+VkzbXotEIb73iqa1foZinNxCy8a7UZDdY/TLZwMK81ofzQrci+K8K1CrCp9AgT5WSKMq3ESRvl5Om956jEId84Ym1UtGsU6qq0WtU1GwU1tqUC8Xirarh/aM9KBwe4ZrTjgK+FitmYIiPlFjZqCQT9eWuSjmczRlLgr6HC2ZgaI+TUMmo7BP0o5wFPdxmjESBX64VvT2SJynh0a0caHIu1pqQ/00FPrUuprwVjKKfdIbWlDhMQp+TDkNKHwTRf96IfaCTqDwHwtgzmcviv8uH942YB7gWtbGYZ7gaMZau/MG3C3Yeu0Z5hE+fYWp4pGYZ3ivGEvOfzAP8ZiTo62Yp7iZoVGYxzicnfquvIaMOsyUjsU8x8cvseI8hXmQ//hxsh7zJFcz0gvzKLuxUf15XsXTKkwEX8U8y0tBPGzHPMytLPTCPM1uDFR8SkfiiR1b167e8PnPp6M91swTffqXLzauXf/Z3nPP6UgqR57faSTzYxvkMKBap5l7oq3Uwz0zO1ULgBzap5GBJ3ypW4pkpuaD3JfpvumWFbqxoVtpyH3hDDJwEXENPXR8C14uP/KXdCuTtndYWfDyDjo89UjLH4l0tvIWAIR2+yHNmqR81yUEvN+aDrwXStkXSGesrwEAENrnoMdquPf1DAFD/Z7Qgf8hrDMSugoMLz3trpW4PbkUGL6ZEOxAVvF4St41DsDecq/bGmTuamoDEzagJK4YVXuQ0Ic2MwBA+eXJ+pe4tAyY0xFPCO4kqidSuglMGzo2Uu/ujQ4G035GCXYlqVg8KW3MA+Db55K+RfRygInbkRJbmKIfkdLUIDMB2Dpe0LOzbW1g6uB0SvAbgrogqT+D2W3tz+vXmdZg+sOkYHtywmJoGWY6AFu363p1pZMNzB9OS3Q+ajYhrZUUAHB8FKVP9/v5gIqv04JriantoSUGFA38V7IeJU30BzXtT2hxv0eK7yWk9UdVAIqszdQf1+rCoOxOWvC8g5IpSOxYdQCqH9Kd/VVB4XBi8BNCyqZQ87ZKAO1v0+BKjntw+9rliHPnzl+8djsq9qmbhpttQOna1Dx/iY4dSOwzh1rgnJ6qRHrkyb3blkwa1LnJu5VLhfmBF51hL1Wt2aTr0ClL//PL6cgMJVKmOkFt/wxi8FsymiC1f4Hy5X8xUebdI9tmDGxWowCY3Vb4tRaD52z/Pcpjor1lQfmz1OCHRPheJWe9egAdoozz3P11+YimFRygvLNy89FrDj0wQWQ7IHALOZccNIQjuSMogNDVbgMe/rq49xtBQGy+mgOXH4o1wL0qBCgcSQ5+TELxZHrqkgBQ86JX7n8/uVlRILxky+l7or0S8R7QWJ+exCIUbEZ6CxABvlPTc5ZyZG7rYsBiqY6L/0rLWfpUXyCyKD24joBX3PQ8BDprnHpBwu6x7/gCq87aE39OfsHpGkBnPD2ZVdXbj/T+SQg4Jqdjyr5xb9iBZZ93Jh5MxfSpDiD0BD34s3LNkOAvKQGo2sAJrAc0qQGkfk0QNlLM5yJFC2ixmvMpumBXqz9SPFQyhlOEfZRy3iephWS0Jemun0pjkOTXJeNdknCkQiGxNJWVjAo0xQSpMx1pDpOMMJpwijIFk2ny2CXDh6jEMFXmIs1JIJrPacKZihRIJipKNmKISsyvxhwk+p5sRBKFM5QIS6LqtmzcoiohnwqzkOqbsnGTKpymQEgCWddkLj7IfOFI9nXZuEMWjjGd3wO6HsjGQ7oifc02EOlOlI14urCvyezXCHPJRgZhl23mao+U+0mGEylvba4/SSsuGcVJO2KqN5H01yTjFdLwNTNtp62JZDSibZuJiqfT1lsy+tKWVtQ8s5H2KZIxlTacYRq/x8RtlYwtxMX4maU7Ev+7ZBwhDrua5Q/qHkjGQ+qOmKQ6kp9fLsKQ/CrmWENfXbn4gL4VpghOpm+UXIyiLyHQDAOR/q1y8R/6sI8Z/mbgmlxcZeCoCSojh4WkoqCHASxv3AIW2khFG+RwtmE+D1lYKRUrWLhvN6o5snhFKq6ygI2N+oYHLC0TZZHHzw0KTmFilEyMZuJpoDG9kMnfZOIIE9jVmF+4cJeQiBJuLnYbUtjFBY6TiHHIZUYBI4YhmxESEcEGDjLiKB8zJWI6HwcNKOpmYxuI5EY2XAW9Nxi5POArE459XGB/7+3n4nIoCGXoJS5+9loBFxOJFUEsy8czkZ7PW32RR3dzEMxGbh6wp7d2MDEFRHMyE995yfmMh59tsmHby0OSr3eaIosPC4NwFrzPAjb0zmoW3A1APOu6WVjunbsszAcBncvCTa9URw4j/CTE9ywHWNkb4Ry43gQRrZHBwcfe2M/BXBDSmRz85AX/VAau+kmJ32UGnvnlriEy2BDEtA4DWC93Cxn4BgT1Mwbm5u4MfU9LSkrxZPpO5KqAh77JIKoT6XPny00bJP9BgKz43yUPW+RmKX0DQVh70bcwN6fIu+wjLfYI8o7nIjSTvI4gru3JcwXlrBlSf9kmL7YL1GGjnM0hrxcIbGfyZubsIHW3fCTG5wZ1+3JkT6JuGIjsEOoSbDmpjsQnBslMYDxxWCUnA6hbDkK7kLp+OdlEnKei1JR1E7chJ+eJ2wdi+wtxZ3Pg7yKuq9x0Ii7D70VvIe2J/nLjF0cbvv6iQcRtBMFdR1z/F60jrrbk1CZu1YuO0xZlkxxbFG1/vsCeQtsaEN2VtD21ZVcRaW8oO/Vow7LZtaXtiUN2HAm0tchuMm1fgPB+Rdsn2X1BWz/p6UfbtuzO0vaS9JSk7UQ29lTSroP4XiftqS1LGSR9nfxsIA1LZWlEWw/56UVbvSxDaKsoPxVoG5hlCWlPQIBjSVuQZSdpv0rQPtK+z3KRtNkSNI+0s1mek9ZFgrqRlgwAhZD0GhJUgzTMD/A6aZl+EuSXSVoNgNakXQMRvkVaC4DhpO2SoV9IGwKwgLSVMrSWtLkAX5AWLkPhpH0G8BtpXWWoK2kHAC6RVkuG6pB2ASCOtNIyVIG0GHB4SAuRofykue0lkHIXCHEGZVjkddIeS1EsaTUak3ZNim6S9mEX0v6RojOkdRhM2u9S9BdpAyaQdkiKDpE2bj5pv0jRL6TNWUfabinaS9rqL0n7QYp2kfb5HtJ2SNFe0nYeJG2vFO0jbf/fpO2TosOk/XGOtMNS9Adpp6+R9rsUHSftSiRpx6ToNGn3Ykk7JUUXSItJIO28FF0hLT6ZtCtSdIu0xOekRUnRY9KepZH2VIrSSUt1kYY+MhSApGd4aCsgQ8Vo87hpKytDlWlzp9H2mgy9S1vqU9rqyVBj2pKf0NZGhjrTFh9D2wAZGkLboyjapsrQbNru36FtnQz9m7Zb12nbLUO/0nb1Em2nZCiCtoiztEXLUDxtZ/6hze2QIH+k/dhB2rCUBJUjbt/3xNWSoLrEfftv4vpK0EfEbVxK3DwJWkLc4qnEfS9Bu4mbMoq4CAm6Stzw3sSl2uXHkUFczzbEYRn5qYjEt6xLXWP5aUld7VepGyc/E6irUZK6L+XnO+qK+biJuyY/t4lz2eERcZ5Q6SmAxEcBnCEO60pPI+pOAOylbqz0TKRuJ8BG6r6Qnu+oWwswg7qb0nOPun8BDKIOS8lOOaS+P0BL8nrJTn/ymgK8Qd4W2dlOXg2AwuTdkZ0o8sIAIIk6LCM5FZH6eACA0+T1lZxB5B3P8jV5X0rOt+R9nmUOeYm+cuNMJm9Glj7kYWO5aY7k98hSi741crORvnezFKEv0iY19kf0FcwCSeTh21JTC8l/Atmepm+u1Cyk70R2X9F3xyYz9vv0fZ7dJPqwnsw0RPo/ya45A9tkZjsDTbIrycCzYIkJec5Asewgjj7sIzH9kf4YeOEhBn6TmKMM7H/RMgawqrxUQwYXv6gPBxvlZTMHPV70GgephaSlcCoH1V/kl8EATpGW6chgmuNFcJ6Dh36y4ozh4DTkcCMH2FtW+iGHa3LSl4VrPpLiuMFCz5xUYgH7ScpAZLFcTiCOhTu+cuK8z8IjyPEeFnCInIxEFnfkbBIPUf5SEhDNQ3jO6vGAE6RkMvL4fs4CXTw8LSkj//Wch3RnzuAUD/iVjHyHPB6HXK5gAutKSANkcmluWnNxwSEfvle4aJabUBcTOF4+JiGT6UG5gb+4SKsqHdXTuTgCuZ7BBZ50yIbvWeRySu5qs4HTZGM2svlu7hzJbGS8IRlvu9hI8Mkd7GEDr4bIReg1ZPNH8OIoPvBbufgR+RzqjaqM4Bip+AQZregNuMeIq7ZM1M9k5BZ4dRUj+LCERJSKQUaXeachJ3guRB7yRSCndb3jSOAE9zmkwe8wchrn4x34nBXcKgy2L5DVbeDlTrzgTFlYgLy29VZwGi84QRKmIa8pgd6Cn5jBcDmYiszuBq9/xA2Ok4KpyG1/7xV1c4PjZWAqcusu7D34jR1cbLN+9lXI7kEwcCA/+Lmv1fP/AfntZ0T+dH7wQIi1C/sD+U0NNQJ2MITnyli5l68gw9+BoR05wriG1q1NEnLc1hj/JI4w8xOLZp/rQY6f+BkDW1hC/DbUihX6FXneCAY3ZArvvG+9mkUj03WMsj9kCjNnO6xVwFrk+p7NKFjMFeKJSlbqrSvI9jwwvBJfmDbVzyqFrMhEtj3ljIMjfCFeqm2N2kUi4/vBhN04Q8/GQtan7E5kvaMZnLGcISZ+4rQ2+ZekI+uPfM0AS3hDvN3ZwviOikPm54MpX+YO8XhTi+LoewO595QzBxxhD/FYUwvi6H8T+d8PJu2mAYjHWtqsRcDg26iDHc3i91iVtDunjv51K4UIxKtDAqxDiblxqIXRvmaBWSqcnNW0pA2yLfnh2K8jKUCMm1PSGryzPQM1cSqYtmia6XZVh1xX/vh3t3qImbtb+ehe2KjzqI0phcwDW8w2FbxbYuIN9RAxalY5jbM3+CIVNXI9mLi6yS7avARga3GYAEQ8PrKInr23Ihq10lPJTLDPXNvAyNpHKEDM3Ne/sGbZas6/hbq5F0zdxFxHDQFoeY0CRHT/Mb68Nvm32vQINfRDc8FFU7krGgN+U9JIyHplRfNA/Xll/L4U1NKzYPL+psKfDAKofJwKREw7NLGWr75UGrg9GrW1l9mcj0yF4UaBY46bjKzPD0+vH6wdzndH/xCDOvvA12ww3lzuHkYBNI6lJGvmhc0fveKjCY7qfdeeSkfdHQ2mD3psKszsYxiUPk9MtqmnNo+oHcpa8Q/HbD2ThjocHWA+mGgu9Ew2DIJ/Jij7qAMrBtcpzk1A9TbhW44loD6PAwWD48yF+HmgUeDYSlX2z859P39Qk0r+1PmWq993+pajDzyo2Y8DVYApZsOIykaBbQVpL/T8GkBb5buo5xNAydAnZsPng40CWMbA3/mA+FLXtSwuWA2YZjrEX6saBWvIOxkK5Je8rWOTQdF8CeZD10J/g2xfEXe9EDBYIVa/noSoAjMUQLz0ijHg9xtpTyoCi7XStWsqKBvyWAVMHWgMhF0nzNMMmBykW9HB6sBIJRDXOAyBykl0zQU2t2vWEFDY94Ya+GuwIdCWrH8cfITc1qprDpWgsyJ4oqAhsJSotMrAaD2PTrUHpW0nFMELhQ3xO0PTdGB1k0b9DYrXUwUvFDQCqqZRdMvJS8F4faqtGvykCp4KMQKmUNQBmB2lTbtA+RpuVfCQnxG+l+g5Btz63tKkzCrqwWZlcLsRUJ+eRuxAb01aDwQWTlAGpxgBn1FzDPj1ualF8QUpgJHqeFoaUfgJMW0ZgqFaNBRI9DmvDCaWNwAG0HLDzlFAvAadsdMAH6iDp50GwF5SxgLLi/THUwuo/EIdXG5E0VhC0gryVFF//gNkFk9Wx9PYAGhLyLfA9B+6k1SUDhivDkblNwA+paMNV4N1ZwwQ6ntZHdxqhOMoFUlOrgpl6s1FByVQ26MONjQAikYR8RWwfVhr3DWB1jUK3fQ3AF5NpqEbX6O1ZiUQG3JPHZxuBDTMoCAzjK+XdeZuEDXQTKHU0kZAdzcBx4HxuxrTGOjdrg5+bQj0dqs3n7Ot+rINCC4Yow7WNAR6uZVrxlk/bXlUgCLootBRY6BdimKeMM4qa0snoHmXOtjMGHgvVq1rwLktUVN2ANHF49Q5YRCUO6fUV6zBYT2JLUYVtFMHmxsEAVtUmsjbp3rSFuj+tzp/GwXQ/Yk6LXkbqCWbgfDgm8pgXcOg+B5lKvBWW0duBFEG72Uq87NxAK1vKpGywcFbEQ1xvQu0z1TGU9kE4Bz/2HRXx4YB98/0YwYQ7ziuCq4zA0DQJ9FmStxUGzTwknYcc1AHFZ6q8iyfKQB8u/1pkifbWztBC/frxtMKQH9vVXCESQCg/NQIo9yn59d1gC5u1Y3ewOEmVS6aBwBeGvT1fW/FH5zXOgx0cpFmbAIW/c8qgu+ZKWuJpuM27bsc58ru2f1TO1eOaloKtPMTvTjrzwNUSFLkC7O9ODB/oYLBDtDWwVqRWB64bK+Iq4wiuttVK9oBn8vUwM2WpJVOLANGff9WI7OqFWmkEX85OIFSsUrgr1akjj7ElgJe67uUwPYW5H1tyKgH3A5V42F+6/GeNgwGftcqgZ9Zj5q6sBoYdhxWArsK1SEHR1DgphJJL4vUjQLAc9UkFTAi2GLU0oKkKsB1C7cKuMcuTu7mwPc4JXCVOI0DzlcqgTMsxfsasBxYt3+nBE4QpW/tvIHzqBI4zULUZu+IE7jPf1EJXG63DB9wF5Ef+P+vKCXwhwAZiiwFOlgjUQk8U1qCEqqBHtZNUwLjmliDOqyl1QFdbO1SAj0r/K1AXc4yWoI+ds5UAvHiuxagHmOZnUAne3vUQPeaUO2rz5e7F+jlYEUQowf7SM1HoJsfq4J4ua1N6z5kazTo5yRlECN6+GhcA64mgo7OVgfxzsQiwjIb9HSWQojpXzfz1bOGPM0EXZ2kEiLGb2rkVMHOXSOWJoK+jlYLEZ//PLqqzTy2St0WHYx/zXp5RoHODvIoljVh/5xW5e0G+VVtN2HLn8mY9Q3uGvPjGQR62ztTvWzTLv6weGzXuhWLBNle5BNW5tU6XcYu+er325mY0ze5a8pOZm/Q3U4ZNOTU8+zRw0cxsXHP0Ntvc9eMm4xOoL+t0ogx/h3umjOT1hp0uE4CL+9x15KXhDqgx9Xus1KTu1as3K8GulwqgpP3uWvNSUQp0Od8RxipzV1bRo7kB512fsNHHe7a8fGtE/Ta9ikbdbnrwMYKO2j3x5lM1OeuIxOZY0DHmyby0IC7TjwkNgU9r3ydhUbcdWXhRmXQ9bD9HDThrhsHB8NA331WMNCcux4MrHaA1g/MIK8ld73IyxgMuv/BI+pacdebusf1QP+LHyWuDXd9iPuzJFhBn4Ue0tpx15e2JQ6wiK2eUNaBu/6UJbYD61jmFGGduBtA2JnyYCWda+nqyt1Aujb6g8XslkRVD+4GU/W0N1jPMn8Q1ZO7oUT9XR6sqH1yBkl9uBtOkmuaD1jUN69S1I+7ERRdfwesa+BaggZwN5qgjUFgaZs/ImcQd2PIiW0DVrfwN9QM5W4cNT8UBQvc5gEtw7kbT0t0B7DG+TZ6KBnJ3QRStoaBZa5/k5DR3E0i5E4jsNIBSzLJGMPdv8hwrwgCi/3WWSrGcTeViohaYL19hj2hIZy76TQkjnaAJS+00U3BBO5mUODZVhQs+9v/EDCJu1kEnK4JVt424LFyU7ibq1z8UDtY/PwrMxSbxt08xTLXFwQBrPCtWjO4W6DWzioghO8cUWkmd4tV+vt9EMQWEerM5m6JOlfbgSza+0WqMpe7papED3GAOAaEx6ixgLtlasRPCQKRDBwbrcJC7parEDsxBMQyYPQD8y3SrpjwIBBN5/BIsy3m7lOzRY8NBPH0G3LHXLO4W2iuqNEBIKI+nY6ZaQh3Y8x0pocvyGnN7zJN8wF3TU3j2V0PhLXMp8nmSPHnLn+mOVLWvQwCm2/8XTN8B+zvN8PDfxUEobU335lpWD3+Ohjm+bW9AyS3xNR7xvwK/NtPGxM9tyyIr735Tpf3EspoALye5j3Pvg6+IMMlplz1UlpD0MKeHi/dnVsWJPnNZQ+9EFsfNLFbihfi139gA2m2N9yalDP3F8VAG6sczkXqt639QKb9O+18lp0nYv7LoJW1Nj98QdqBvqEg2baS7zZt1ejVYNDQknU7dO9Qr7wP/H///z9XAg=='


def PrintAsBase64PNGString(mol, highlightAtoms=[], molSize = (200, 200)):
    data = Draw._moltoimg(mol,molSize,highlightAtoms,"",returnPNG=True, kekulize=True)
    return b64encode(data).decode('ascii')

def nan_NA(v):
    if v=='N/A':
        return np.nan
    else:
        return float(v)
    
app = dash.Dash(__name__)
server = app.server

mols = [m for m in Chem.SDMolSupplier('molecules.sdf', removeHs=False)]
df = pd.DataFrame([m.GetPropsAsDict() for m in tqdm(mols)])
df=df.assign(mol=mols)
df.CB1_pKi = df.CB1_pKi.apply(nan_NA)
df.CB2_pKi = df.CB2_pKi.apply(nan_NA)
print('Rendering structures...')

try:
    imgs = np.load('imgs_back.npy')
except Exception as e:
    print(e)
    imgs = [PrintAsBase64PNGString(m) for m in tqdm(df.mol.values)]
    np.save('imgs_back', imgs)
    
df=df.assign(imgs=imgs)

slider = dcc.RangeSlider(id='slider', min=0, max=10, step=0.1, value=[0,10],
                             marks={str(n):{'label':'%.2f'%n} for n in np.arange(11)})

app.layout = html.Div([dcc.Location(id='url', refresh=False),
        html.Table([
                html.Tr(html.Td(html.B('Select pKi range:'))),
                html.Tr(html.Td(slider, style={'width':'25%',
                                               'padding-right':'10em',
                                               'padding-left':'10em'})),
                html.Tr(html.Td([
                        html.Td(html.Button('Show', id='button', style={'width':'15%','margin-top':'2.5em'}), 
                                style={'width':'85%','margin-top':'2.5em'}),
                        html.Td([
                                html.A(html.Img(src=git_icon, style={'width':'15px','height':'15px'}),
                                       href='https://github.com/MikolajMizera/SMViewer', target='blank'),
                                       html.A('VSMolView', href='https://github.com/MikolajMizera/SMViewer', 
                                              target='blank')], 
                                              style={'padding-left':'10em',
                                                     'margin-top':'2.5em',
                                                     'width':'15%'})]))
                ], style={'width':'100%'}),
        dcc.Loading(html.Table([],
                               id='results',
                               style={'width':'100%',
                                      'border-spacing':'1em',
                                      'margin-top':'2em',
                                      'border-top-width':'1px',
                                      'border-top-style':'dashed',
                                      'border-bottom-style':'dashed',
                                      'border-top-color':'grey'}))])

@app.callback(
dash.dependencies.Output('slider', 'marks'),
[dash.dependencies.Input('url', 'pathname')],
[dash.dependencies.State('url', 'pathname')])
def update_ranges(_, pathname):
    
    if pathname == '/CB1':
        r = 'CB1'
    elif pathname == '/CB2':
        r = 'CB2'
    else:
        return {('%.2f'%n):{'label':'%.2f'%n} for n in np.linspace(0,10, 10)}
        
    pkis = df['%s_pKi'%r].dropna().values
    
    return {('%.2f'%n):{'label':'%.2f'%n} for n in np.linspace(pkis.min(), pkis.max(), 10)}

@app.callback(
dash.dependencies.Output('slider', 'min'),
[dash.dependencies.Input('url', 'pathname')],
[dash.dependencies.State('url', 'pathname')])
def update_ranges(_, pathname):
    
    if pathname == '/CB1':
        r = 'CB1'
    elif pathname == '/CB2':
        r = 'CB2'
    else:
        return 0
    
    pkis = df['%s_pKi'%r].dropna().values
        
    return pkis.min()

@app.callback(
dash.dependencies.Output('slider', 'max'),
[dash.dependencies.Input('url', 'pathname')],
[dash.dependencies.State('url', 'pathname')])
def update_ranges(_, pathname):
    
    if pathname == '/CB1':
        r = 'CB1'
    elif pathname == '/CB2':
        r = 'CB2'
    else:
        return 10
        
    pkis = df['%s_pKi'%r].dropna().values
    
    return pkis.max()

@app.callback(
dash.dependencies.Output('slider', 'value'),
[dash.dependencies.Input('url', 'pathname')],
[dash.dependencies.State('url', 'pathname')])
def update_ranges(_, pathname):
    
    if pathname == '/CB1':
        r = 'CB1'
    elif pathname == '/CB2':
        r = 'CB2'
    else:
        return [0,10]
        
    pkis = df['%s_pKi'%r].dropna().values
    
    return [pkis.min(), pkis.max()]
    
    
@app.callback(
dash.dependencies.Output('results', 'children'),
[dash.dependencies.Input('button', 'n_clicks')],
[dash.dependencies.State('slider', 'value'),
  dash.dependencies.State('url', 'pathname')])

def update_table(n_clicks, pki_range, pathname):
    if not n_clicks:
        return []
    
    if pathname == '/CB1':
        r = 'CB1'
    elif pathname == '/CB2':
        r = 'CB2'
    else:
        return []
        
    pki_min, pki_max = pki_range
    
    df_r = df.dropna(subset=['%s_pKi'%r])
    mask = (df_r['%s_pKi'%r]>=pki_min) & (df_r['%s_pKi'%r]<=pki_max)
    masked_df = df_r[mask].sort_values(by='%s_pKi'%r).iloc[:250]
    
    imgs = [html.Img(src='data:image/png;base64,%s'%img) for img in masked_df.imgs]
    labels = [html.P(['pKi (%s): %.2f'%(r, pki), 
                      html.Br(), 
                      html.A('%s'%chid, 
                             href='https://www.ebi.ac.uk/chembl/compound_report_card/%s/'%chid,
                             target='blank')],
                        style={'text-align': 'center'}) for chid, pki in zip(masked_df['Molecule ChEMBL ID'].values,
              masked_df['%s_pKi'%r].values)]
    tds = [html.Td([html.Tr(i), html.Tr(l)]) for i,l in zip(imgs, labels)]
    inds = np.array_split(np.arange(len(tds)), int(np.ceil(len(tds)/4)))
    trs = [html.Tr([tds[i] for i in ind]) for ind in inds]    
    
    return trs

    

if __name__ == '__main__':
    app.run_server(debug=False)