# Labeling guide

This corpus is synthetic. Every message was written for VA Mail Sorter. None of it is a real letter, a real veteran, or a copy of an agency template. Names, claim numbers, Social Security numbers, and account numbers are omitted on purpose. Mailboxes use `example.com`, `example.org`, or `example.net` only.

The sorter is a triage aid. It does not give legal or benefits advice, and a label in this file is not a decision about anyone's benefits.

## Labels

Each row has:

| Column | Meaning |
| --- | --- |
| `label` | `ACTION_NEEDED` or `INFORMATIONAL` |
| `suspicious` | `1` if the message is a phishing look-alike, otherwise `0` |
| `hard_case` | `none`, `phishing`, `urgent_newsletter`, or `polite_deadline` |
| `notice_type` | The kind of notice the text is modeled on |
| `source` | `VA`, `DFAS`, or `VGLI` |
| `split` | `train`, `val`, or `test` |

`SUSPICIOUS` is not a third training class. It is a flag. At triage time it overrides the class: the report shows `SUSPICIOUS` instead of the action or informational prediction. The class label underneath is whatever the look-alike is imitating.

## When the label is ACTION_NEEDED

The fictional reader would be worse off for ignoring the message. A date, a payment, a missing page, or a visit confirmation is doing real work in the letter.

- Benefit decision that still has a review or disagreement window.
- Debt or overpayment, including waiver and payment-plan choices.
- VGLI premium due, and lapse or grace-period warnings.
- Appointment that should be confirmed or rescheduled.
- Claim development that asks for evidence by a date.
- Address verification where mail or payment can pause.
- myPay direct deposit that was returned and needs a new instruction.
- Polite deadline notices. The tone is soft. The date still matters. `hard_case` is `polite_deadline`.

## When the label is INFORMATIONAL

Nothing is due. Ignoring the message does not create a fictional deadline.

- Benefit decision or award confirmation with no review window and an explicit statement that no reply is needed.
- Appointment hours, parking, booth moves, and after-visit summaries that do not book a visit.
- Claim status that says the file was received or is still processing, with no evidence request.
- myPay or 1099-R availability. Downloading a statement is optional in this dataset; the letter sets no tax-filing deadline.
- Premium receipt after a payment posted.
- Newsletters and optional surveys.
- Urgent-sounding newsletters. Words like "urgent" and "act now" are bait. The body says it is a newsletter, that no action is required, and how to unsubscribe. `hard_case` is `urgent_newsletter` and the label stays `INFORMATIONAL`.

## Suspicious override

`suspicious` is `1` only for phishing look-alikes, and those rows use `hard_case` `phishing`.

- `phishing_demand` imitates a benefit suspension, a debt, or a premium threat and asks for a password, a Social Security number, a one-time code, a gift card, a wire, or bitcoin. The class label is `ACTION_NEEDED` because that is the demand it imitates. The flag overrides it.
- `phishing_lure` imitates a statement-ready note. The class label is `INFORMATIONAL`. A look-alike login URL is why the flag is on.

Cues were chosen because they are not how VA, DFAS, or a VGLI servicer collects a real debt or delivers a real statement: credentials in email, gift cards, wire or bitcoin, "account will be deleted", and hosts that contain `login`, `verify-now`, or `secure-update`.

## What we refused to put in a row

- A real person's name, a real claim number, a Social Security number, or an account number.
- Any email or URL whose host is not under `example.com`, `example.org`, or `example.net`.
- A verbatim agency letter. The situations are the public ones (decisions, debts, premiums, appointments, claim status, newsletters, surveys, tax-statement availability, address checks), and the sentences are original.

## Splits

`scripts/build_dataset.py` assigns `train` / `val` / `test` at about 60/20/20, stratified on `label` and `suspicious`, with `random_state` 42.

- Model parameters and cross-validation use `train` only.
- The class threshold and the suspicious cutoff use `val` only.
- `test` is held out. It is not used to edit rules, choose C, or move a threshold.

Do not retune after reading test errors. If a row is mislabeled against this guide, fix the label and regenerate from the script rather than editing the test file by hand.
